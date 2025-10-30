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
import numpy as np
import torch
from scene import Scene
import sys
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, modified_render
import torchvision
from utils.general_utils import safe_state, multi_yaml_parsing_render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams, MaskingParams
from gaussian_renderer import GaussianModel
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import contextlib
from train_stage1 import set_render_func

def render_set(model_path, name, iteration, views, gaussians, render_func, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    kwargs = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_dict = render_func(view, gaussians, pipeline, background, **kwargs)
        rendering = render_dict["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, 
                iteration : int,
                opt : OptimizationParams, 
                pipeline : PipelineParams, 
                mask_params : MaskingParams, 
                args):
    
    render_train = args.render_train
    render_interpolation = args.render_interpolation
    render_extrapolation = args.render_extrapolation
    render_explore1 = args.render_explore1
    render_explore2 = args.render_explore2
    
    dataset.dataset_distribution = (
        "train" if render_train else
        "interpolation" if render_interpolation else
        "extrapolation" if render_extrapolation else
        "explore1" if render_explore1 else
        "explore2" if render_explore2 else None
    )
    
    # with torch.no_grad():
    render_func = set_render_func(opt)
    no_grad = not (mask_params.use_fisher or mask_params.use_visibility_mask)
    with torch.no_grad() if no_grad else contextlib.nullcontext():
        print("[DEBUG] : dataset = ", dataset)
        print("[DEBUG] : opt = ", opt)
        print("[DEBUG] : pipeline = ", pipeline)
        
        gaussians = GaussianModel(dataset, opt, mask_params)
        
        print("load iteration = ", iteration if iteration != -1 else "auto search of max iteration")
            
        scene = Scene(dataset, mask_params, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_configs = []
        if render_train:
            render_configs.append(("train", scene.getTrainCameras()))
        if render_interpolation:
            render_configs.append(("interpolation", scene.getTestCameras()))
        if render_extrapolation:
            render_configs.append(("extrapolation", scene.getTestCameras()))
        if render_explore1:
            render_configs.append(("explore1", scene.getTestCameras()))
        if render_explore2:
            render_configs.append(("explore2", scene.getTestCameras()))

        # Fisher rendering
        if mask_params.use_fisher:
            from masking.fisher_info import render_fisher_set
            print("[INFO] : render fisher-uncertainty")
            for name, cameras in render_configs:
                render_fisher_set(dataset.model_path, name, scene.loaded_iter, cameras, scene.getTestCameras(), gaussians, modified_render, pipeline, background)

        # Visibility rendering
        if mask_params.use_visibility:
            from masking.visibility import render_visibility
            print("[INFO] : render visibility custom")
            for name, cameras in render_configs:
                render_visibility(dataset.model_path, name, scene.loaded_iter, cameras, scene.getTestCameras(), gaussians, render_func, pipeline, background)

        # Visibility mask rendering
        if mask_params.use_visibility_mask:
            from masking.visibility_mask import render_visibility_mask
            print("[INFO] : render visibility mask ; ExtraNeRF style")
            
            # refer utils/virtual_cam_utils.py - prepare_diffusion_cameras
            pair_cameras = []
            test_cameras = sorted(scene.getTestCameras(), key=lambda cam: cam.image_name)
            train_cameras = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)
            pair_cameras.append(train_cameras)
                
            render_visibility_mask(dataset.model_path, "test", scene.loaded_iter, test_cameras, pair_cameras, 
                                  gaussians, render_func, pipeline, background)

        # Scale mask rendering
        if mask_params.use_scale_mask:
            from masking.scale_mask import render_scale_mask
            print("[INFO] : render scale mask ; 3DGS-Enhancer style")
            
            for name, cameras in render_configs:
                render_scale_mask(dataset.model_path, name, scene.loaded_iter, cameras, gaussians, render_func, pipeline, background)

        # View direction mask rendering
        if mask_params.use_viewdirection_mask:
            from masking.viewdirection_mask import render_viewdirection_mask
            print("[INFO] : render viewdirection mask ; 3DGS-Enhancer style")
            render_viewdirection_mask(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene.getTrainCameras(), gaussians, render_func, pipeline, background)
            
        # Original RGB rendering
        if not args.skip_original:
            print("[INFO] : render original RGB")
            for name, cameras in render_configs:
                render_set(dataset.model_path, name, scene.loaded_iter, cameras, gaussians, render_func, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    mask_params = MaskingParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_original", action="store_true")
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--render_explore1", action="store_true")
    parser.add_argument("--render_explore2", action="store_true")
    parser.add_argument("--render_interpolation", action="store_true")
    parser.add_argument("--render_extrapolation", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config", "-c", type=str, default="")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--logname", type=str, default="") # only for logging at monitoring server
    # FIXME: Current config codes ignore cfg_args
    parser.add_argument("--override_fisher", action="store_true")
    parser.add_argument("--override_visibility", action="store_true")
    parser.add_argument("--override_visibility_mask", action="store_true")
    parser.add_argument("--override_scale_mask", action="store_true")
    parser.add_argument("--override_viewdirection_mask", action="store_true")
    parser.add_argument("--config_system", "-cs", type=str, default="")
    parser.add_argument("--config_gs", "-cg", type=str, default="")
    parser.add_argument("--config_diffusion", "-cd", type=str, default="")
    parser.add_argument("--config_uncertainty", "-cu", type=str, default="")
    parser.add_argument("--config_virtualcam", "-cv", type=str, default="")
    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--stage1", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    # args override
    args.use_fisher = True if args.override_fisher else False
    args.use_visiblity = True if args.override_visibility else False
    args.use_visibility_mask = True if args.override_visibility_mask else False
    args.use_scale_mask = True if args.override_scale_mask else False
    args.use_viewdirection_mask = True if args.override_viewdirection_mask else False
    if not args.stage1:
        scene_name = os.path.basename(args.source_path)
        save_expname = multi_yaml_parsing_render(args)
        save_expname = f"stage2_{args.expname}/{save_expname}" # HARDCODED
        args.model_path = os.path.join("output", save_expname, scene_name)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    for dist_name in ["explore1", "explore2", "interpolation", "extrapolation"]:
        args.dataset_distribution = dist_name
        print(f"[INFO] : render {dist_name}")
        render_sets(model.extract(args), args.iteration, optimization.extract(args), pipeline.extract(args), mask_params.extract(args), args)
