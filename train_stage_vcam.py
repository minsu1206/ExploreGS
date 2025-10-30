"""
stage2 : extract virtual cameras and save.
"""
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import copy
import random
from utils.loss_utils import l1_loss, ssim, get_expon_lr_func
from gaussian_renderer import network_gui, modified_render, render_original, render_upgrade
import sys
from scene import Scene, GaussianModel, SpaceSearch
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
# from utils.camera_utils import CameraGenerator
# from utils.vmf_utils import kappa_logscale_clamp, compute_loss_uncertainty, uncertainty_control, save_mask_rendered_images
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, DiffusionParams, MaskingParams, VirtualCamParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import torchvision
import numpy as np
import lpips

# stage2
from typing import List
import time
from utils.virtual_cam_utils import CameraGenerator, export_cameras, load_camera_metadata, save_camera_metadata
from diffusion.general_utils import convert_cam_to_diff_img, convert_cam_to_diff_pose
from diffusion.enhance_utils import EnhanceDiffusionPriorPipeline
from diffusion.depth_utils import DepthExtractor
from utils.general_utils import render_based_on_mask_type
from utils.confidence_utils import ConfidenceScore
import einops
import gc
import glob
from PIL import Image

def calc_loss_func(pred: torch.Tensor, viewpoint_cam, opt, diff_params, crop_params=None, confidence_dict=None):
    """
    Calculate loss with support for confidence-aware training.
    :param pred: Predicted tensor.
    :param viewpoint_cam: Camera object with original images.
    :param opt: Options containing lambda_dssim, pixel_confidence, and image_confidence.
    :param diff_params: Parameters defining cropping or resizing behavior.
    :param crop_params: 2-length list defining crop bounds.
    :param confidence_dict: Optional dictionary for confidence weights.
    :return: Loss and L1 term.
    """
    # TODO: develop
    def apply_confidence(term, confidence_key, use_confidence):
        if use_confidence and confidence_dict is not None:
            if confidence_key in confidence_dict:
                return (term * confidence_dict[confidence_key]).mean()
        return term.mean()

    def compute_l1_ssim_terms(pred, gt_image, mean=True, crop_bounds=None):
        if crop_bounds:
            pred_crop = pred[:, crop_bounds[0]:crop_bounds[2], crop_bounds[1]:crop_bounds[3]]
            gt_crop = gt_image[:, crop_bounds[0]:crop_bounds[2], crop_bounds[1]:crop_bounds[3]]
        else:
            pred_crop = pred
            gt_crop = gt_image

        l1_term = l1_loss(pred_crop, gt_crop, mean=mean)
        ssim_term = 1.0 - ssim(pred_crop, gt_crop)
        return l1_term, ssim_term

    px_confidence = opt.pixel_confidence
    img_confidence = opt.image_confidence

    if px_confidence or img_confidence != "":
        if confidence_dict is None and viewpoint_cam.virtual:
            raise ValueError("[WARNING]: calc_loss_func: confidence_dict is None but pixel_confidence or image_confidence is True")

    gt_image = viewpoint_cam.original_image.cuda()

    if viewpoint_cam.virtual:
        if crop_params is not None and diff_params.crop == "squares":
            gt_image2 = viewpoint_cam.extra_image.cuda()
            repeat_box = (crop_params[1][0], crop_params[1][1], crop_params[0][2], crop_params[0][3])

            # Compute L1 and SSIM terms
            Ll1, ssim1 = compute_l1_ssim_terms(pred, gt_image, crop_bounds=crop_params[0], mean=not px_confidence)
            Ll2, ssim2 = compute_l1_ssim_terms(pred, gt_image2, crop_bounds=crop_params[1], mean=not px_confidence)
            Ll_repeat, ssim_repeat = compute_l1_ssim_terms(pred, gt_image, crop_bounds=repeat_box, mean=not px_confidence)

            # Combine L1 and SSIM terms
            l1_term = Ll1 + Ll2 - Ll_repeat
            l1_term = apply_confidence(l1_term, "px_confidence", px_confidence)
            ssim_term = (1.0 - ssim1) + (1.0 - ssim2) - (1.0 - ssim_repeat)

            loss = (1.0 - opt.lambda_dssim) * l1_term + opt.lambda_dssim * ssim_term
            loss = apply_confidence(loss, "img_confidence", img_confidence != "")
            return loss, l1_term

        elif diff_params.crop in ["center", "force_resize", "raw"]:
            if diff_params.crop == "center":
                pred = center_crop(pred)
            if diff_params.crop == "raw":
                # resize to be same shape as gt_image
                pred = F.interpolate(pred[None], size=(gt_image.shape[1], gt_image.shape[2]), mode="bilinear")[0]

            Ll1, ssim_term = compute_l1_ssim_terms(pred, gt_image, mean=not px_confidence)
            Ll1 = apply_confidence(Ll1, "px_confidence", px_confidence)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_term
            loss = apply_confidence(loss, "img_confidence", img_confidence != "")
            return loss, Ll1
        else:
            raise NotImplementedError(f"Crop type '{diff_params.crop}' is not implemented.")

    else:
        Ll1, ssim_term = compute_l1_ssim_terms(pred, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_term
        return loss, Ll1

def prepare_output_and_logger(args, mask_params):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    merged_args = {**vars(args), **vars(mask_params)}
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        # cfg_log_f.write(str(Namespace(**vars(args))))
        cfg_log_f.write(str(Namespace(**merged_args))) # MaskParams should be also saved 

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training(dataset, opt, pipe, diff_params, mask_params, cam_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger=None, args=None):
    # Settings -------------------------------------------------------------- #
    first_iter = 0
    # else:
    #     tb_writer = None
    if args.swap:
        dataset.dataset_distribution = "swap"
    
    tb_writer = prepare_output_and_logger(args, mask_params)
    tb_writer = None

    # gaussians = GaussianModel(dataset.sh_degree)
    gaussians = GaussianModel(dataset, opt, mask_params)
    scene = Scene(dataset, mask_params, gaussians)

    if opt.appearance_model == "original":
        render_func = render_original
        print("[DEBUG] : using original render")
    elif opt.appearance_model == "upgrade":
        render_func = render_upgrade
    else:
        render_func = render_upgrade # no use of render ; deprecated
        print("[DEBUG] : using default render")
    
    early_stop = 300 if args.earlystop else False
    
    # time.sleep(60)
    
    image_log_interval = args.image_log_interval
    
    if early_stop:
        print(f"[INFO] : training early stops at {early_stop}")
    
    print(f"[CUDA] : before load gaussian ckpt {(torch.cuda.memory_allocated()/1024**3):.3f}") # [CUDA] : before load gaussian ckpt 0.056 [15/02 04:15:36] 
    print(f"[CUDA] : before load gaussian ckpt {(torch.cuda.memory_reserved()/1024**3):.3f}") # [CUDA] : before load gaussian ckpt 0.117 [15/02 04:15:36]
    
    if checkpoint:
        print("[INFO] : loading checkpoint ", checkpoint)
        (model_params, ckpt_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, mask_params)
    else:
        raise ValueError("No checkpoint found")
    
    del model_params
    print(f"[CUDA] : after del gaussian ckpt {(torch.cuda.memory_allocated()/1024**3):.3f}") # [CUDA] : after del gaussian ckpt 12.510 [15/02 05:18:41]   
    print(f"[CUDA] : after del gaussian ckpt {(torch.cuda.memory_reserved()/1024**3):.3f}") # [CUDA] : after del gaussian ckpt 14.480 [15/02 05:18:41]

    candidate_views = []
    
    # space_search = SpaceSearch(scene, gaussians, dataset, opt, pipe, render_func, modified_render, diff_params, mask_params, cam_params)
    # candidate_views = space_search.search()
    # candidate_views = space_search.connect_cameras(candidate_views)
    
    # exit()
    # virtual_trajector = space_search.build_trajectory(candidate_views)
    # virtual_trajector = space_search.build_trajectory()
    # candidate_views = space_search.get_all_trajectory() # for visualization
    # candidate_views += space_search.get_all_rejected()
    # print(f"[DEBUG] : candidate_views : {len(candidate_views)}")
    # exit()
    
    first_iter = 0
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # CameraGenerator load -------------------------------------------------- #
    virtual_cam_generator = CameraGenerator(scene, cam_params, diff_params, mask_params,
                                            renderer=render_func, gaussians=gaussians, pipe=pipe, opt=opt, bg=background, args=args)
    virtual_cam_dataset = []
    
    # time.sleep(6)
    # ----------------------------------------------------------------------- #

    space_search = SpaceSearch(scene, gaussians, virtual_cam_generator, dataset, opt, pipe, render_func, modified_render, diff_params, mask_params, cam_params)
    virtual_cam_dataset = space_search.build()
    
    print(f"[CUDA] : after space search {(torch.cuda.memory_allocated()/1024**3):.3f}")
    print(f"[CUDA] : after space search {(torch.cuda.memory_reserved()/1024**3):.3f}")
    # [CUDA] : after space search 5.437 [15/02 04:17:55] # 실제로는 한 4.8GB 찍히는듯
    # [CUDA] : after space search 13.832 [15/02 04:17:55]
    
    # Save result of search
    export_cameras(scene.getTrainCameras() + scene.getTestCameras() + virtual_cam_dataset + candidate_views, 
                   f"{scene.model_path}/{args.config_virtualcam.split('/')[-1][:-3]}.json")
    del space_search
    # del virtual_cam_generator
    
    torch.cuda.empty_cache()
    gc.collect()
    # ----------------------------------------------------------------------- #
    # Apply diffusion result
        # raise ValueError(f"[ERROR] : dataset_update_counts = {cam_params.dataset_update_counts} \
        #                     & dataset_update_percentage = {cam_params.dataset_update_percentage} but \
        #                     len(virtual_cam_dataset) = {len(virtual_cam_dataset)}")
    n_updates = len(virtual_cam_dataset)

    vcam_dir = os.path.join(scene.model_path, "vcam")
    os.makedirs(vcam_dir, exist_ok=True)
    vcam_meta_dir = os.path.join(vcam_dir, "metadata")
    vcam_img_dir = os.path.join(vcam_dir, "images")
    vcam_dbg_dir = os.path.join(vcam_dir, "debugging")
    vcam_gen_dir = os.path.join(vcam_dir, "gen_virtual")
    
    os.makedirs(vcam_meta_dir, exist_ok=True)
    os.makedirs(vcam_img_dir, exist_ok=True)
    os.makedirs(vcam_dbg_dir, exist_ok=True)
    os.makedirs(vcam_gen_dir, exist_ok=True)
    
    iteration = 9999
    print(f"[CUDA] : before prepare_diffusion_cameras {(torch.cuda.memory_allocated()/1024**3):.3f}")
    print(f"[CUDA] : before prepare_diffusion_cameras {(torch.cuda.memory_reserved()/1024**3):.3f}")
    kwargs_pkg = {"iteration": iteration, "mask_params": mask_params, "pipe": pipe, "bg": background, "gaussians": gaussians, "scene": scene, "threshold": diff_params.binarized_threshold}
    diff_input_pkg = virtual_cam_generator.prepare_diffusion_cameras(virtual_cam_dataset, n_updates, **kwargs_pkg)
    diff_input_pkg["aux_ref_pose"] = torch.from_numpy(virtual_cam_generator.train_cameras_all_trans)
    
    # save diff_input_pkg
    torch.save(diff_input_pkg, os.path.join(vcam_dir, "diff_input_pkg.pth"))
    
    for cam in virtual_cam_dataset:
        try:
            save_camera_metadata(cam, os.path.join(vcam_meta_dir, f"{cam.image_name}.json"))
        except Exception as e:
            import pdb; pdb.set_trace()
    
    print("[INFO] : train_stage_vcam.py : done")
    
def multi_yaml_parsing(args):
    save_name = ""
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for cfg_p in [args.config_system, args.config_gs, args.config_diffusion, args.config_uncertainty, args.config_virtualcam]:
        # save_name += cfg_p.split("/")[-2][:3] + "_"
        # save_name += cfg_p.split("/")[-1] + "-"
        cfg = OmegaConf.load(cfg_p + ".yaml")
        for k in cfg.keys():
            recursive_merge(k, cfg)
            
    save_name = ""
    save_name += f"vcam_{args.config_virtualcam.split('/')[-1]}/"
    
    save_name += f"diffusion_{args.config_diffusion.split('/')[-1]}/"

    save_name += f"gs_{args.config_gs.split('/')[-1]}/"

    start_checkpoint_path = args.start_checkpoint.split('/')[:-1]
    for path in start_checkpoint_path:
        if "stage1" in path:
            save_name += f"from_{path}"
            break

    return args, save_name

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = DiffusionParams(parser)
    cp = VirtualCamParams(parser)
    mp = MaskingParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.2")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[9_000, 12_000, 15_000, 18_000, 21_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--logname", type=str, default="") # just for logging at monitoring server ; do not anything at all.
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config_system", "-cs", type=str, default="")
    parser.add_argument("--config_gs", "-cg", type=str, default="")
    parser.add_argument("--config_diffusion", "-cd", type=str, default="")
    parser.add_argument("--config_uncertainty", "-cu", type=str, default="")
    parser.add_argument("--config_virtualcam", "-cv", type=str, default="")
    parser.add_argument("--expname", type=str, required=True)
    parser.add_argument("--image_log_interval", type=int, default=1000)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--earlystop", action="store_true")
    # parser.add_argument("--no_sds", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    scene_name = os.path.basename(args.source_path)
    # scene_expname
    # if args.config != "":
    #     save_expname = ""
    #     if os.path.exists(args.config):
    #         # TODO:
    #         save_expname = args.expname
    #         cfg = OmegaConf.load(args.config)
    #         def recursive_merge(key, host):
    #             if isinstance(host[key], DictConfig):
    #                 for key1 in host[key].keys():
    #                     recursive_merge(key1, host[key])
    #             else:
    #                 assert hasattr(args, key), key
    #                 setattr(args, key, host[key])
    #         for k in cfg.keys():
    #             recursive_merge(k, cfg)
    #     else:
    #         print(f"[WARNING] : train.py : {args.config} file does not exist")
    #         exit()

    #     args.model_path = os.path.join("output", args.expname, scene_name) if save_expname == "" else os.path.join("output", save_expname, scene_name)
    
    args, save_expname = multi_yaml_parsing(args)
    # iter_num = args.start_checkpoint.split("chkpnt")[-1].split(".")[0]
    # save_expname = f"stage2_{iter_num}/{save_expname}"
    # save_expname = save_expname + "_" + args.expname if args.expname != "" else save_expname
    save_expname = f"stage2_{args.expname}/{save_expname}"
    
    args.model_path = os.path.join("output", save_expname, scene_name)
    if args.swap:
        args.model_path = os.path.join("output", save_expname, scene_name + "_swap")
    print("Optimizing " + args.model_path)
    logger = wandb.init(name=f"stage2_{args.expname}_{scene_name}", project='3dgs-diff')
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), dp.extract(args), mp.extract(args), cp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
            logger=logger, args=args)

    # All done
    print("\nTraining complete.")
    wandb.finish()

