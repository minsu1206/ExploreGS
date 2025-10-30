"""
stage1 : vanilla 3DGS training
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
from random import randint
from utils.loss_utils import l1_loss, ssim, get_expon_lr_func
from gaussian_renderer import network_gui, render_original, render_upgrade
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, MaskingParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torchvision.transforms import ToPILImage
import torchvision
import numpy as np

def set_render_func(opt):
    app_model = opt.appearance_model
    use_invdepth = opt.inv_depth # always True
    if app_model == "expmaps" or use_invdepth:
        print("[INFO] : using upgraded-3DGS render")
        return render_upgrade
    elif app_model == "original":
        print("[INFO] : using original render")
        return render_original
    elif app_model == "hier":
        raise NotImplementedError("hier render not implemented")
    else:
        print("[INFO] : using default render")
        return render

def training(dataset, opt, pipe, mask_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger=None, args=None):
    # Settings -------------------------------------------------------------- #
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, mask_params)
    # else:
    #     tb_writer = None
    tb_writer = None

    print(f"[INFO] : args.swap = {args.swap}")
    
    if args.swap:
        dataset.dataset_distribution = "swap"

    gaussians = GaussianModel(dataset, opt, mask_params)
    # gaussians = GaussianModelHybrid(dataset, opt, mask_params) if opt.appearance_model == "featsplat" else GaussianModel(dataset, opt, mask_params)
    scene = Scene(dataset, mask_params, gaussians)
    render_func = set_render_func(opt)
    print(render_func)
    
    early_stop = 10000 if args.earlystop else False
    
    image_log_interval = args.image_log_interval
    
    if early_stop:
        print(f"[INFO] : training early stops at {early_stop}")

    gaussians.training_setup(opt, mask_params)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, mask_params)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Depth regularization ; from updated 3DGS
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_L1depth_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # For debugging
        # import time
        # time.sleep(iteration*2)
        # if iteration > 50:
        #     break
        
        if early_stop:  # quite hardcoded
            if iteration > 1000:
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

        gaussians.update_learning_rate(iteration)
        gaussians.update_sh_degree(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:   # TODO: slowing down maybe helpful
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # viewpoint_cam = viewpoint_stack[0]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_func(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["invdepth"][0, :, :][None] # accumulated depth
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            # exit()
                    
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
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
                                      image_log_interval, scene.model_path, dataset, mask_params)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            # Optimizer step
            if iteration < opt.iterations:
                if opt.appearance_model == "expmaps":
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # print(f"[DEBUG] : GPU MEM CHECK : {iteration:04d}th iteration after optim.step = {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                # gaussians.vmf_mu_normalize() # only act when use_vmf = True
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, opt, mask_params):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    merged_args = {**vars(args), **vars(opt), **vars(mask_params)}
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
        # "train_loss_patches/sds_loss": sds_loss.item(),
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
                    # render_dict = renderFunc(viewpoint, scene.gaussians, pipe=renderArgs[0], bg_color=renderArgs[1], vmf_evaluate=False)
                    render_dict = renderFunc(viewpoint, scene.gaussians, pipe=renderArgs[0], bg_color=renderArgs[1])
                    image = torch.clamp(render_dict["render"], 0.0, 1.0)           
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    use_depth = viewpoint.invdepthmap is not None
                    if use_depth:
                        invdepth = torch.clamp(render_dict["invdepth"][0, :, :][None], 0.0, 1.0)
                        gt_invdepth = torch.clamp(viewpoint.invdepthmap.cuda(), 0.0, 1.0)
                    prefix = config["name"] + "-render-" + viewpoint.image_name

                    if iteration >= 100:
                        torchvision.utils.save_image(image, os.path.join(render_path, prefix + ".png"))
                        if use_depth:   
                            torchvision.utils.save_image(invdepth, os.path.join(render_path, prefix + "-invdepth.png"))
                        # save_mask_rendered_images(mask_params, render_dict, render_path, prefix, viewpoint)

                    # skipping gt
                    if iteration == image_log_interval:
                        torchvision.utils.save_image(gt_image, os.path.join(render_path, "gt-" + viewpoint.image_name + ".png"))
                        if use_depth:
                            torchvision.utils.save_image(gt_invdepth, os.path.join(render_path, "gt-invdepth-" + viewpoint.image_name + ".png"))
                    
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mp = MaskingParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 15_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config", "-c", type=str, default="")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--image_log_interval", type=int, default=5000)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--earlystop", action="store_true")
    # parser.add_argument("--no_sds", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    scene_name = os.path.basename(args.source_path)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    save_expname = "stage1_"
    save_expname += args.config.split("/")[-1]
    cfg = OmegaConf.load(args.config + ".yaml")
    for k in cfg.keys():
        recursive_merge(k, cfg)
    args.model_path = os.path.join("output", save_expname, scene_name)
    if args.swap:
        args.model_path = os.path.join("output", save_expname, scene_name + "_swap")
    print("Optimizing " + args.model_path)
    logger = wandb.init(name=f"stage1_{args.expname}_{scene_name}", project='3dgs-diff')
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), mp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
            logger=logger, args=args)

    # All done
    print("\nTraining complete.")
    wandb.finish()
