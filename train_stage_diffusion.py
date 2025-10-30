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
import einops
import gc
import glob
from PIL import Image
import shutil

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

def training(dataset, opt, pipe, diff_params, mask_params, cam_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, 
             logger=None, args=None, override_vcam=""):
    # Settings -------------------------------------------------------------- #
    first_iter = 0
    # else:
    #     tb_writer = None
    tb_writer = prepare_output_and_logger(args, mask_params)
    model_path = args.model_path
    
    if args.swap:
        dataset.dataset_distribution = "swap"
    
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
    
    first_iter = 0
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # Diffusion model load -------------------------------------------------- #
    if diff_params.prior_type == "enhancer":
        diffusion_pipeline = EnhanceDiffusionPriorPipeline(diff_params, args, pipe, model_path) # 4GB+
    
    print(f"[CUDA] : after load diffusion {(torch.cuda.memory_allocated()/1024**3):.3f}")
    print(f"[CUDA] : after load diffusion {(torch.cuda.memory_reserved()/1024**3):.3f}")
    
    vcam_dir = os.path.join(model_path, "vcam")
    os.makedirs(vcam_dir, exist_ok=True)
    
    if override_vcam != "":
        vcam_meta_dir = os.path.join(override_vcam, "vcam", "metadata")
    else:
        vcam_meta_dir = os.path.join(vcam_dir, "metadata")
    print("[DEBUG] : vcam_meta_dir ", vcam_meta_dir)
    os.makedirs(vcam_meta_dir, exist_ok=True)
    # vcam_meta_dir = os.path.join(vcam_dir, "metadata")
    vcam_img_dir = os.path.join(vcam_dir, "images")
    vcam_dbg_dir = os.path.join(vcam_dir, "debugging")
    vcam_gen_dir = os.path.join(vcam_dir, "gen_virtual")
    os.makedirs(vcam_gen_dir, exist_ok=True)
    os.makedirs(vcam_img_dir, exist_ok=True)
    os.makedirs(vcam_dbg_dir, exist_ok=True)
    
    iteration = 9999
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"[CUDA] : after prepare_diffusion_cameras {(torch.cuda.memory_allocated()/1024**3):.3f}")
    print(f"[CUDA] : after prepare_diffusion_cameras {(torch.cuda.memory_reserved()/1024**3):.3f}")
    
    if override_vcam != "":
        diff_input_pkg = torch.load(os.path.join(override_vcam, "vcam", "diff_input_pkg.pth"), map_location="cpu")
    else:
        diff_input_pkg = torch.load(os.path.join(vcam_dir, "diff_input_pkg.pth"), map_location="cpu")
    gen_images = diffusion_pipeline.generate(diff_input_pkg, save_path=vcam_gen_dir)
    print("[DEBUG] : gen_images ; ", gen_images.shape)
    # gen_images = gen_images.copy()
    # del diffusion_pipeline
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[CUDA] : after del diffusion {(torch.cuda.memory_allocated()/1024**3):.3f}")
    print(f"[CUDA] : after del diffusion {(torch.cuda.memory_reserved()/1024**3):.3f}")
    
    gaussians = GaussianModel(dataset, opt, mask_params)
    scene = Scene(dataset, mask_params, gaussians)
    if checkpoint:
        print("[INFO] : loading checkpoint ", checkpoint)
        (model_params, ckpt_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, mask_params)
    else:
        raise ValueError("No checkpoint found")
    
    
    virtual_cam_dataset = []
    vcam_name_set = diff_input_pkg["names"]
    
    for vcam_names in vcam_name_set:
        for vcam_name in vcam_names:
            if "virtual" not in vcam_name:
                virtual_cam_dataset.append(None) # dummy ; never used
            else:
                vcam_meta_path = os.path.join(vcam_meta_dir, f"{vcam_name}.json")
                cam = load_camera_metadata(vcam_meta_path)
                virtual_cam_dataset.append(cam)
                
                if override_vcam != "":
                    target_dir = os.path.join(model_path, 'vcam', 'metadata')
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(vcam_meta_path, target_dir)
    # medata_jsons = sorted(glob.glob(os.path.join(vcam_meta_dir, "*.json")))
    # if len(medata_jsons) == 0:
    #     raise ValueError(f"[ERROR] : no meta json found in {vcam_meta_dir}")
    
    # for meta_json in medata_jsons:
    #     cam = load_camera_metadata(meta_json)
    #     virtual_cam_dataset.append(cam)
    
    # write index combinations for debugging
    updated_idxs = diff_input_pkg["idx_set"]
    
    print("[DEBUG] : updated_idxs ", updated_idxs)
    print("[DEBUG] : gen_images ", gen_images.shape)
    
    with open(f"{vcam_gen_dir}/comb.txt", "w") as f:
        for idx, update_id in enumerate(updated_idxs):
            f.write(f"{virtual_cam_dataset[update_id].image_name}\n")
    
    for idx, up_idx in enumerate(updated_idxs): # to handle
        # if virtual_cam_dataset[up_idx].last_iter != iteration: # to prevent double update
        virtual_cam_dataset[up_idx].last_iter = iteration
        # virtual_cam_dataset[up_idx].original_image = gen_images[idx]
        virtual_cam_dataset[up_idx].original_image = gen_images[up_idx]

    # check match
    
    for idx, up_idx in enumerate(updated_idxs):
        with torch.no_grad():
            render_pkg = render_func(virtual_cam_dataset[up_idx], scene.gaussians, pipe, background)
        torchvision.utils.save_image(render_pkg["render"], os.path.join(vcam_dbg_dir, f"{virtual_cam_dataset[up_idx].image_name}_render.png"))
        torchvision.utils.save_image(virtual_cam_dataset[up_idx].original_image, os.path.join(vcam_dbg_dir, f"{virtual_cam_dataset[up_idx].image_name}_original.png"))

    # Extract depth map from virtual camera
    if opt.guided_sampling or opt.depth_reg_gen:
        print("[DEBUG] : depth extraction")
        
        depth_extractor = DepthExtractor(dataset)
        # depth extraction
        visibility_filter_ordered = einops.rearrange(diff_input_pkg["visibility_filters"], "b t n -> (b t) n")
        pixel_gs_counter_ordered = einops.rearrange(diff_input_pkg["pixel_gs_counters"], "b t h w -> (b t) h w")
        invdepth_ordered = einops.rearrange(diff_input_pkg["invdepths"], "b t h w -> (b t) h w")
        avgdepth_ordered = einops.rearrange(diff_input_pkg["avgdepths"], "b t h w -> (b t) h w")
        renders_ordered = einops.rearrange(diff_input_pkg["renders"], "b t c h w -> (b t) c h w")
        
        print("[DEBUG] : visibility_filter_ordered ", visibility_filter_ordered.shape)
        print("[DEBUG] : pixel_gs_counter_ordered ", pixel_gs_counter_ordered.shape)
        print("[DEBUG] : invdepth_ordered ", invdepth_ordered.shape)
        print("[DEBUG] : avgdepth_ordered ", avgdepth_ordered.shape)
        print("[DEBUG] : renders_ordered ", renders_ordered.shape)
        
        for idx, up_idx in tqdm(enumerate(updated_idxs), desc=f"Extract depth / Guided Sampling = {opt.guided_sampling}"):
            depth_input_pkg = {"gaussians": gaussians, "camera": virtual_cam_dataset[up_idx], 
                                "visibility_filter": visibility_filter_ordered[up_idx], 
                                "pixel_gs_counter": pixel_gs_counter_ordered[up_idx],
                                "image": gen_images[up_idx],
                                "invdepth": invdepth_ordered[up_idx],
                                "avgdepth": avgdepth_ordered[up_idx]}
            depth, scale_info = depth_extractor.extract_depth(depth_input_pkg) # depth = (H,W)
            virtual_cam_dataset[up_idx].set_depthmap(depth, scale_info)

    shutil.rmtree(vcam_meta_dir)
    os.makedirs(vcam_meta_dir, exist_ok=True)
    use_depth = False
    # for cam in virtual_cam_dataset:
    for idx in updated_idxs:
        cam = virtual_cam_dataset[idx]
        # if cam.last_iter == iteration:  # train_cam_index -> omit
        torchvision.utils.save_image(cam.original_image, os.path.join(vcam_img_dir, f"{cam.image_name}_rgb.png"))
        if cam.invdepthmap is not None:
            torchvision.utils.save_image(cam.invdepthmap, os.path.join(vcam_img_dir, f"{cam.image_name}_depth.png"))
            use_depth = True
        save_camera_metadata(cam, os.path.join(vcam_meta_dir, f"{cam.image_name}.json"))
    
    print("[INFO] : train_stage_diffusion.py : done")

def get_override_vcam(override_vcam, args):
    if override_vcam == "":
        return ""

    save_name = ""
    for cfg_p in [args.config_system, args.config_gs]:
        save_name += cfg_p.split("/")[-2][:3] + "_"
        save_name += cfg_p.split("/")[-1] + "-"
        
    save_name += "dif_"
    save_name += override_vcam + "-"
    
    for cfg_p in [args.config_uncertainty, args.config_virtualcam]:
        save_name += cfg_p.split("/")[-2][:3] + "_"
        save_name += cfg_p.split("/")[-1] + "-"

    return save_name[:-1]

from train_stage_vcam import multi_yaml_parsing

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[12_000, 13_000, 15_000, 17_000, 20_000, 30_000])
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
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--image_log_interval", type=int, default=1000)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--earlystop", action="store_true")
    parser.add_argument("--override_vcam", type=str, default="")
    parser.add_argument("--lustre", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("[INFO] : args.override_vcam : ", args.override_vcam)
    scene_name = os.path.basename(args.source_path)
    
    args, save_expname = multi_yaml_parsing(args)
    # iter_num = args.start_checkpoint.split("chkpnt")[-1].split(".")[0]
    # save_expname = f"stage2_{iter_num}/{save_expname}"
    # save_expname = save_expname + "_" + args.expname if args.expname != "" else save_expname
    if args.expname != "":
        save_expname = f"stage2_{args.expname}/{save_expname}"
    else:
        print("[WARNING] : ambiguous expname")
        iter_num = args.start_checkpoint.split("chkpnt")[-1].split(".")[0]
        save_expname = f"stage2_{iter_num}/{save_expname}"
    args.model_path = os.path.join("output", save_expname, scene_name)
    if args.swap:
        args.model_path = os.path.join("output", save_expname, scene_name + "_swap")
    print("Optimizing " + args.model_path)
    logger = wandb.init(name=f"stage2_{args.expname}_{scene_name}", project='3dgs-diff')
    # Initialize system state (RNG)
    safe_state(args.quiet)

    override_vcam = get_override_vcam(args.override_vcam, args)
    if override_vcam != "":
        override_vcam = os.path.join("output", f"stage2_{args.expname}", override_vcam, scene_name)
        if args.swap:
            override_vcam += "_swap"
        if args.lustre:
            override_vcam = override_vcam.replace("output", "/workspace/dataset_lustre/users/minsu/ckpt")
            print("[INFO] : from lustre : override_vcam : ", override_vcam)
    print("[INFO] : override vcam : ", override_vcam)
    
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), dp.extract(args), mp.extract(args), cp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
            logger=logger, args=args, override_vcam=override_vcam)

    # All done
    print("\nTraining complete.")
    wandb.finish()

