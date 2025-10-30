"""
stage2 : leverage diffusion prior for extrapolation
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
import glob
from PIL import Image
from utils.virtual_cam_utils import load_camera_metadata
from utils.loss_utils import l1_loss, ssim, get_expon_lr_func
from gaussian_renderer import network_gui, modified_render, render_original, render_upgrade
import sys
from scene import Scene, GaussianModel, SpaceSearch
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
# from utils.camera_utils import CameraGenerator
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
from utils.confidence_utils import ConfidenceScore

def calc_loss_func(pred: torch.Tensor, viewpoint_cam, opt, diff_params, crop_params=None, confidence_dict=None, loss_dict=None):
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
        if crop_params is not None and diff_params.crop == "squares": # Deprecated
            raise NotImplementedError("crop_params is not None and diff_params.crop == 'squares' is not implemented")

        elif diff_params.crop in ["center", "force_resize", "raw"]:
            if diff_params.crop == "center":
                pred = center_crop(pred)
            if diff_params.crop == "raw":
                # resize to be same shape as gt_image
                pred = F.interpolate(pred[None], size=(gt_image.shape[1], gt_image.shape[2]), mode="bilinear")[0]

            # print("[DEBUG] : pred shape : ", pred.shape, "gt_image shape : ", gt_image.shape)
            # print("[DEBUG] : pred.dtype : ", pred.dtype, "gt_image.dtype : ", gt_image.dtype)
            
            # Ll1, ssim_term = compute_l1_ssim_terms(pred, gt_image, mean=not px_confidence)
            # Ll1 = apply_confidence(Ll1, "px_confidence", px_confidence)
            # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_term
            # loss = apply_confidence(loss, "img_confidence", img_confidence != "")
            # return loss, Ll1
            
            if opt.virtual_loss == "dinov2": # feature leve loss
                loss = 1 - loss_dict["loss"]
                # TODO: apply confidence
                Ll1, ssim_term = compute_l1_ssim_terms(pred.detach(), gt_image)
                Ll1 = Ll1.detach() # dummy
            elif opt.virtual_loss == "lpips":
                loss = loss_dict["loss"] # (1,1,H,W)
                # TODO: apply confidence
                loss = apply_confidence(loss, "px_confidence", px_confidence)
                loss = apply_confidence(loss, "img_confidence", img_confidence != "")
                Ll1, ssim_term = compute_l1_ssim_terms(pred.detach(), gt_image)
                Ll1 = Ll1.detach() # dummy. no use
            else: # pixel level loss
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

def training(dataset, opt, pipe, diff_params, mask_params, cam_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, 
             logger=None, args=None, override_vcam=""):
    # Settings -------------------------------------------------------------- #
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, mask_params)
    # else:
    #     tb_writer = None
    tb_writer = None
    if args.swap:
        dataset.dataset_distribution = "swap"

    # gaussians = GaussianModel(dataset.sh_degree)
    gaussians = GaussianModel(dataset, opt, mask_params)
    scene = Scene(dataset, mask_params, gaussians)

    if opt.appearance_model == "original":
        render_func = render_original
        print("[DEBUG] : using original render")
    elif opt.appearance_model == "upgrade":
        render_func = render_upgrade
    else:
        render_func = render_upgrade
        print("[DEBUG] : using default render") # no use of render ; deprecated
    
    early_stop = 300 if args.earlystop else False
    
    # time.sleep(60)
    
    image_log_interval = args.image_log_interval
    
    if early_stop:
        print(f"[INFO] : training early stops at {early_stop}")
    
    if checkpoint:
        print("[INFO] : loading checkpoint ", checkpoint)
        (model_params, ckpt_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, mask_params)
    else:
        raise ValueError("No checkpoint found")

    first_iter = 0
    # print(f"[CUDA] : after load gaussian ckpt {(torch.cuda.memory_allocated()/1024**3):.3f}")
    # print(f"[CUDA] : after load gaussian ckpt {(torch.cuda.memory_reserved()/1024**3):.3f}")
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    confidence_score = ConfidenceScore(args, opt, scene, pipe, mask_params, background, render_func)
    # time.sleep(6)
    # ----------------------------------------------------------------------- #

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    update_all_once = False # iterate all virtual cam at least once

    ema_loss_for_log = 0.0
    ema_L1depth_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if checkpoint_iterations == []:
        checkpoint_iterations = [int(opt.iterations // 2), opt.iterations]

    # ----------------------------------------------------------------------- #
    
    print("[INFO] : load virtual cam dataset")
    virtual_cam_dataset = []
    if override_vcam == "":
        vcam_dir = os.path.join(scene.model_path, "vcam")
    else:
        print("[INFO] : override vcam : ", override_vcam)
        vcam_dir = os.path.join(override_vcam, "vcam")
    
    vcam_meta_dir = os.path.join(vcam_dir, "metadata")
    vcam_img_dir = os.path.join(vcam_dir, "images")
    if not os.path.exists(vcam_meta_dir):
        raise ValueError(f"[ERROR] : vcam_meta_dir does not exist : {vcam_meta_dir}")
    if not os.path.exists(vcam_img_dir):
        raise ValueError(f"[ERROR] : vcam_img_dir does not exist : {vcam_img_dir}")
    
    use_depth = opt.guided_sampling or opt.depth_reg_gen
    
    medata_jsons = glob.glob(os.path.join(vcam_meta_dir, "*.json"))
    if len(medata_jsons) == 0:
        raise ValueError(f"[ERROR] : no meta json found in {vcam_meta_dir}")
    
    for meta_json in medata_jsons:
        cam = load_camera_metadata(meta_json)
        original_img_path = os.path.join(vcam_img_dir, f"{cam.image_name}_rgb.png")
        cam.original_image = torch.tensor(np.array(Image.open(original_img_path))).permute(2, 0, 1).float() / 255.0 # (H, W, C) -> (C, H, W)
        if use_depth:
            depth_img_path = os.path.join(vcam_img_dir, f"{cam.image_name}_depth.png")
            depth_img = torch.tensor(np.array(Image.open(depth_img_path).convert("L"))).float() / 255.0  # (H, W, C) -> (C, H, W)
            cam.set_depthmap(depth_img, None)
        virtual_cam_dataset.append(cam)
    
    crop_params = []
    
    # ----------------------------------- #

    dataset_update_iter = args.dataset_update_iter
               
    for iteration in range(first_iter, opt.iterations + 1):

        if early_stop:  # quite hardcoded
            if iteration - first_iter > early_stop:
                break

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_func(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        if not update_all_once:
            valid_virtual_view_stacks = [cam for cam in virtual_cam_dataset if cam.last_iter > 0]
            merged_viewpoints = viewpoint_stack + valid_virtual_view_stacks
            if len(valid_virtual_view_stacks) == len(virtual_cam_dataset):
                update_all_once = True
                print("[INFO] : set update_all_once as True")
                
                if pipe.exit_update_all_once:
                    exit()
        else:
            merged_viewpoints = viewpoint_stack + virtual_cam_dataset
        viewpoint_cam = merged_viewpoints[random.randint(0, len(merged_viewpoints)- 1)] 
        # print("[DEBUG] : viewpoint_cam.image_name : ", viewpoint_cam.image_name)

        # Render
        # if (iteration - 1) == debug_from:
        #     pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_func(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss # TODO: fix for n-squares version
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        additional_loss = {}
        # print(f"[DEBUG] : image shape : {image.shape} / viewpoint_cam : {viewpoint_cam.image_width} x {viewpoint_cam.image_height}")
        
        # TODO: get confidence_dict
        if viewpoint_cam.virtual:
            conf_dict_pkg = {"opt": opt, "scene": scene, "pipe": pipe, "mask_params": mask_params, "bg": bg, "render_func": render_func,
                             "viewpoint_cam": viewpoint_cam, "train_cameras": scene.getTrainCameras(), "gaussians": gaussians,
                             "diff_params": diff_params, "iteration": iteration, "gen": viewpoint_cam.original_image, "image": image}
            confidence_dict = confidence_score.get_confidence_dict(conf_dict_pkg)
            loss_dict = confidence_score.get_loss_dict(conf_dict_pkg)
        else:
            confidence_dict = None
            loss_dict = None
        
        loss, Ll1 = calc_loss_func(pred=image, viewpoint_cam=viewpoint_cam, opt=opt, 
                                   diff_params=diff_params,
                                   crop_params=crop_params,
                                   confidence_dict=confidence_dict if viewpoint_cam.virtual else None,
                                   loss_dict=loss_dict)
        
        # loss, additional_loss = compute_loss_uncertainty(mask_params, render_pkg, opt, iteration, 
        #                                                 gt_image, image, gaussians, viewpoint_cam, 
        #                                                 l1_loss, loss, additional_loss)

        if opt.lambda_depth_dist > 0:
            lambda_dist = opt.lambda_depth_dist if iteration > 3000 else 0.0
            depth = render_pkg["depth"]
            distortion = depth[2, :, :]
            loss += lambda_dist * distortion.mean()
        
        if opt.lambda_depth_gap > 0:
            lambda_depth_gap = opt.lambda_depth_gap if iteration > 3000 else 0.0
            depth = render_pkg["depth"]
            depth_gap = (torch.nan_to_num((depth[1, :, :] - depth[0, :, :]) / depth[0, :, :], 0, 0)).abs().mean()
            loss += lambda_depth_gap * depth_gap
        
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["invdepth"][0, :, :][None] # accumulated depth
            
            if viewpoint_cam.virtual:
                if use_depth:
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()
                else:
                    mono_invdepth = None
                    Ll1depth = 0
            else:
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
            # exit()
            
            if mono_invdepth is not None:
                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
            # print("[DEBUG] : depth loss is disabled", depth_l1_weight(iteration) > 0 , viewpoint_cam.depth_reliable)
        
        loss.backward()
    
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_L1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_L1depth_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_L1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_func, (pipe, background))
            else:
                training_report_wandb(logger, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_func, (pipe, background), 
                                      image_log_interval, scene.model_path, dataset, mask_params, additional_loss=additional_loss)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            densification_iter = iteration + ckpt_iter
            if densification_iter < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if densification_iter > opt.densify_from_iter and densification_iter % opt.densification_interval == 0:
                    size_threshold = 20 if densification_iter > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)
                
                if densification_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and densification_iter == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            # Optimizer step
            if iteration < opt.iterations:
                if not pipe.freeze_gs: # freeze gaussians to observe diffusion only
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # print(f"[DEBUG] : GPU MEM CHECK : {iteration:04d}th iteration after optim.step = {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    print("[INFO] : train_stage_finetune.py : done")

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def training_report_wandb(logger, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, image_log_interval, model_path, dataset, mask_params, additional_loss=None):
    report_dict = {
        "train_loss_patches/l1_loss":Ll1.item(),
        "train_loss_patches/total_loss": loss.item(),
        "iter_time":elapsed
    }
    logger.log(report_dict, step=iteration)

    if iteration % image_log_interval == 0:
        render_path = os.path.join(model_path, f"log-{iteration:04d}")
        os.makedirs(render_path, exist_ok=True)
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # render_dict = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    render_dict = renderFunc(viewpoint, scene.gaussians, pipe=renderArgs[0], bg_color=renderArgs[1])
                    image = torch.clamp(render_dict["render"], 0.0, 1.0)           
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # print(f"[DEBUG] : image .shape = {image.shape} / gt_image .shape = {gt_image.shape} / viewpoint.hw = {viewpoint.image_height, viewpoint.image_width}")
                    # too heavy
                    # wandb.log({f"{config['name']}_view/{viewpoint.image_name}/render":wandb.Image(image[None])}, step=iteration)
                    # if iteration == testing_iterations[0]:
                    #     wandb.log({f"{config['name']}_view/{viewpoint.image_name}/ground_truth":
                    #                 wandb.Image(gt_image[None])}, step=iteration)
                    # offline logging
                    # image resizing
                    prefix = config["name"] + "-render-" + viewpoint.image_name
                    if iteration >= 100:
                        torchvision.utils.save_image(image, os.path.join(render_path, prefix + ".png"))
                        # save_mask_rendered_images(mask_params, render_dict, render_path, prefix, viewpoint)

                    # skipping gt
                    if iteration == image_log_interval:
                        torchvision.utils.save_image(gt_image, os.path.join(render_path, "gt-" + viewpoint.image_name + ".png"))
                    
                    if iteration in testing_iterations:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    else:
                        if idx > 10: break  # prevent logging getting too heavy memory 

                if iteration in testing_iterations:
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])          
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    wandb.log({f"{config['name']}/loss_viewpoint_l1-loss":l1_test}, step=iteration)
                    wandb.log({f"{config['name']}/loss_viewpoint_psnr":psnr_test}, step=iteration)
        wandb.log({"train/scene_opacity": wandb.Histogram(np_histogram=np.histogram(scene.gaussians.get_opacity.cpu().numpy()))}, step=iteration)
        wandb.log({"train/total_points": scene.gaussians.get_xyz.shape[0]}, step=iteration)
        torch.cuda.empty_cache()

# def multi_yaml_parsing(args):
#     save_name = ""
#     def recursive_merge(key, host):
#         if isinstance(host[key], DictConfig):
#             for key1 in host[key].keys():
#                 recursive_merge(key1, host[key])
#         else:
#             assert hasattr(args, key), key
#             setattr(args, key, host[key])
#     for cfg_p in [args.config_system, args.config_gs, args.config_diffusion, args.config_uncertainty, args.config_virtualcam]:
#         save_name += cfg_p.split("/")[-2][:3] + "_"
#         save_name += cfg_p.split("/")[-1] + "-"
#         cfg = OmegaConf.load(cfg_p + ".yaml")
#         for k in cfg.keys():
#             recursive_merge(k, cfg)
#     return args, save_name[:-1]

from train_stage_vcam import multi_yaml_parsing

def get_override_vcam(override_vcam, args):
    if override_vcam == "":
        return ""

    save_name = ""
    save_name += args.config_system.split("/")[-2][:3] + "_"
    save_name += args.config_system.split("/")[-1] + "-"
    
    save_name += "gs_"
    save_name += override_vcam + "-"
    
    save_name += "dif_"
    save_name += "enhancer4-"
    
    for cfg_p in [args.config_uncertainty, args.config_virtualcam]:
        save_name += cfg_p.split("/")[-2][:3] + "_"
        save_name += cfg_p.split("/")[-1] + "-"
        
    return save_name[:-1]
    
def compute_patchwise_lpips(img1_tensor, img2_tensor, patch_size, loss_fn):
    """
    Compute LPIPS loss patchwise between two images
    Args:
        img1_tensor: (B,C,H,W) tensor normalized to [-1,1]
        img2_tensor: (B,C,H,W) tensor normalized to [-1,1] 
        patch_size: (h,w) tuple for patch dimensions
        loss_fn: LPIPS model instance
    Returns:
        patch_losses: List of LPIPS values for each patch
        mean_loss: Average LPIPS across patches
    """
    _, _, H, W = img1_tensor.shape
    patch_h, patch_w = patch_size
    
    patch_losses = 0
    
    for i in range(0, H - patch_h + 1, patch_h):
        for j in range(0, W - patch_w + 1, patch_w):
            patch1 = img1_tensor[:, :, i:i+patch_h, j:j+patch_w]
            patch2 = img2_tensor[:, :, i:i+patch_h, j:j+patch_w]
            
            # with torch.no_grad():
            patch_losses += loss_fn(patch1, patch2)
            # patch_losses.append(patch_lpips.item())
    
    return patch_losses

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 6_000, 9_000, 12_000, 15_000, 18_000, 21_000, 24_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 15_000, 30_000])
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
    parser.add_argument("--image_log_interval", type=int, default=15000)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--earlystop", action="store_true")
    parser.add_argument("--override_vcam", type=str, default="")
    parser.add_argument("--lustre", action="store_true")
    # parser.add_argument("--no_sds", action="store_true")
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

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    override_vcam = get_override_vcam(args.override_vcam, args)
    if override_vcam != "":
        override_vcam = os.path.join("output", f"stage2_{args.expname}", override_vcam, scene_name)
        if args.swap:
            override_vcam += "_swap"
        if args.lustre:
            override_vcam = override_vcam.replace("output", "/workspace/dataset_lustre/users/minsu/ckpt")
            print("[INFO] : from lustre : override_vcam : ", override_vcam)
    print("[INFO] : override vcam : ", override_vcam)
    if not os.path.exists(override_vcam):
        print(f"[WARNING] : override vcam {override_vcam} does not exist")
    
    training(lp.extract(args), op.extract(args), pp.extract(args), dp.extract(args), mp.extract(args), cp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
            logger=logger, args=args, override_vcam=override_vcam)

    # All done
    print("\nTraining complete.")
    wandb.finish()

