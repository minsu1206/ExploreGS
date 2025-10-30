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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import copy

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.cuda.manual_seed(0)
    # turn on for reproducibility
    torch.cuda.manual_seed_all(0)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def render_based_on_mask_type(iteration, query_cameras, pair_cameras, gaussians, scene, pipe, mask_params, bg, renderer, threshold=0.5, save_mode=True):
    if mask_params.use_fisher:
        from masking.fisher_info import render_fisher_set
        return render_fisher_set(scene.model_path, "virtual", iteration, 
                                  query_cameras, pair_cameras,
                                  copy.deepcopy(gaussians), renderer, pipe, mask_params, bg, save_mode=save_mode)
    elif mask_params.use_visibility:
        from masking.visibility import render_visibility
        return render_visibility(scene.model_path, "virutal", iteration,
                                 query_cameras, pair_cameras, 
                                 copy.deepcopy(gaussians), renderer, pipe, bg, save_mode=save_mode)
    elif mask_params.use_viewdirection_mask:
        from masking.viewdirection_mask import render_viewdirection_mask
        return render_viewdirection_mask(scene.model_path, "virutal", iteration,
                                 query_cameras, pair_cameras, 
                                 copy.deepcopy(gaussians), renderer, pipe, bg, save_mode=save_mode)
    elif mask_params.use_scale_mask:
        from masking.scale_mask import render_scale_mask
        return render_scale_mask(scene.model_path, "virutal", iteration,
                                 query_cameras, 
                                 copy.deepcopy(gaussians), renderer, pipe, bg, save_mode=save_mode)
    elif mask_params.use_visibility_mask:
        from masking.visibility_mask import render_visibility_mask
        return render_visibility_mask(scene.model_path, "virutal", iteration,
                                 query_cameras, [pair_cameras], 
                                 copy.deepcopy(gaussians), renderer, pipe, bg, save_mode=save_mode, threshold=threshold)
    else:
        raise NotImplementedError("No valid rendering method specified.")
    

def multi_yaml_parsing_render(args):
    save_name = ""
    
    save_name += f"vcam_{args.config_virtualcam.split('/')[-1]}/"
    
    save_name += f"diffusion_{args.config_diffusion.split('/')[-1]}/"

    save_name += f"gs_{args.config_gs.split('/')[-1]}/"
    
    start_checkpoint_path = args.start_checkpoint.split('/')[:-1]
    for path in start_checkpoint_path:
        if "stage1" in path:
            save_name += f"from_{path}"

    return save_name
