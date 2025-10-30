import torch
import os
from tqdm import tqdm
import itertools
from einops import reduce, repeat
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torchvision 
import numpy as np
import torch.nn.functional as F
import copy

_EPS = 1e-9
_INF = 1e10

def capture(gaussians):
    return (
        gaussians.active_sh_degree,
        gaussians._xyz,
        gaussians._features_rest,
        gaussians._features_dc,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians.max_radii2D,
        gaussians.xyz_gradient_accum,
        gaussians.denom)

@torch.no_grad()
def render_scale_mask(model_path,
                    name,
                    loaded_iter, 
                    query_cameras,
                    gaussians, 
                    renderer, 
                    pipeline,
                    background, 
                    save_mode=True, 
                    threshold=0.5,
                    vis_cam_index:int=None):
    """
    3DGS-Enhancer style : volume rendering with scale value
    """
    render_path = os.path.join(model_path, f"log-{loaded_iter:04d}", f"scale_mask")    
    if save_mode:
        os.makedirs(render_path, exist_ok=True)
        print("[INFO] : render_scale_mask : save at ", render_path)

    params = list(capture(gaussians)[1:7])
    params = [p.requires_grad_(False) for i, p in enumerate(params)]
    n_gs = len(gaussians.get_xyz)
    # opacity = gaussians.get_opacity.detach().requires_grad_(False)
    scale = gaussians.get_scaling.detach().requires_grad_(False)
    
    original_scales = []
    for idx, cam in tqdm(enumerate(query_cameras), desc="query-cameras"):
        render_pkg = renderer(cam, gaussians, pipeline, background, override_color=scale)
        scale_img = render_pkg["render"]
        scale_img = scale_img[0] * scale_img[1] * scale_img[2]
        scale_img = scale_img ** (1/3)
        scale_img = scale_img.clamp(0, 1)
        original_scales.append(scale_img)
        
    original_scales = torch.stack(original_scales, dim=0)
    min_scale = original_scales.min()
    max_scale = original_scales.max()
        
    minmax_scaled_scales = (original_scales - min_scale) / (max_scale - min_scale)
        
    for idx, cam in tqdm(enumerate(query_cameras), desc="query-cameras"):
        scale_img = original_scales[idx]
        save_path = os.path.join(render_path, f"{cam.image_name}_scale.png")
        scaled_scale_img = minmax_scaled_scales[idx]
        if not os.path.exists(save_path):
            torchvision.utils.save_image(scale_img, save_path)
            torchvision.utils.save_image(scaled_scale_img, save_path.replace("_scale.png", "_minmax_scale.png"))
    
    # break
    if vis_cam_index is not None:
        vis_cam = query_cameras[vis_cam_index]
        render_pkg = renderer(cam, gaussians, pipeline, background)
        T = render_pkg["T"]
        gs_counter = render_pkg["gaussians_pixel_counter"]
        T = torch.nan_to_num(T / gs_counter, nan=0.0)
        
        vis_gs = copy.deepcopy(gaussians)
        
        print(vis_gs._features_dc.shape)
        print(vis_gs._features_rest.shape)
        
        vis_gs._features_rest = torch.zeros_like(vis_gs._features_rest)
        
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.rainbow  # You can choose any colormap you like
        colors = cmap(norm(T.cpu().numpy()))[:, :3]  # Exclude alpha channel if present

        vis_gs._features_dc = torch.from_numpy(colors).float().to(T.device).unsqueeze(1)
        save_path = os.path.join(render_path, f"{vis_cam.image_name}_T.ply")
        vis_gs.save_ply(save_path)
        
        T_magnitude = torch.norm(T_cache_mean, dim=1)  # Shape: (N,)

        # Normalize T_magnitude between 0 and 1
        norm = plt.Normalize(vmin=0, vmax=T_magnitude.max().item())
        cmap = plt.cm.rainbow  # Choose your colormap

        # Apply colormap
        colors = cmap(norm(T_magnitude.cpu().numpy()))[:, :3]  # Shape: (N, 3)

        # Convert to tensor and assign to features_dc
        vis_gs._features_dc = torch.from_numpy(colors).float().to(T.device).unsqueeze(1)
        print(vis_gs._features_dc.shape)  # Should be (N, 1, 3)
        save_path = os.path.join(render_path, f"{vis_cam.image_name}_T_mean.ply")
        vis_gs.save_ply(save_path)
        
    return scaled_scale_img # do not use raw scale image