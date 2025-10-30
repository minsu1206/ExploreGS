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
def render_viewdirection_mask(model_path,
                           name,
                           loaded_iter, 
                           query_cameras,   # test
                           pair_cameras,    # train
                           gaussians, 
                           renderer, 
                           pipeline,
                           background, 
                           save_mode=True, 
                           threshold=0.5,
                           vis_cam_index:int=None):
    """
    (1) closest train camera pose
    (2) compute cosine similarity between test camera pose and (1)
    """

    # Debugging
    # query_c2w_forward = torch.stack([cam.c2w[:3, 2] for cam in query_cameras])
    # pair_c2w_forward = torch.stack([cam.c2w[:3, 2] for cam in pair_cameras])
    
    # cos_sim = query_c2w_forward.unsqueeze(1) * pair_c2w_forward.unsqueeze(0)
    # cos_sim = cos_sim.sum(dim=-1)
    # print("[DEBUG] : cos_sim.shape = ", cos_sim.shape)
    # cos_sim_max = cos_sim.max(dim=-1)[0]
    # print(cos_sim_max.shape)
    # print("[DEBUG] : cos_sim.min = ", cos_sim.min(), "cos_sim.max = ", cos_sim.max())
    # print("[DEBUG] : cos_sim_max.min = ", cos_sim_max.min(), "cos_sim_max.max = ", cos_sim_max.max(), cos_sim_max)
    # import time
    # time.sleep(3)
    # exit()

    render_path = os.path.join(model_path, 
                               f"log-{loaded_iter:04d}", 
                               f"viewdirection_mask")
    if save_mode:
        os.makedirs(render_path, exist_ok=True)
        print("[INFO] : render_visibility_mask : save at ", render_path)

    params = list(capture(gaussians)[1:7])
    params = [p.requires_grad_(False) for i, p in enumerate(params)]
    # (1) test cameras -> mark visible gaussians
    # (2) render at nearest train cameras and track them
    # implementation note : https://www.notion.so/Visibility-Mask-3DGS-1318db77e98780e78c40e2c5054a8145
    n_gs = len(gaussians.get_xyz)
    opacity = gaussians.get_opacity.detach().requires_grad_(False)
    
    direction_cache = torch.zeros((len(pair_cameras), n_gs, 3), device=opacity.device) # (K, #_GS) but K = len(all_train_cams) / save forward vector
    # visible_counters = torch.zeros(n_gs, device=opacity.device)
    
    for idx, pair_cam in tqdm(enumerate(pair_cameras), desc="pair-cameras"):
        render_pkg = renderer(pair_cam, gaussians, pipeline, background)
        # T = render_pkg["T"] # (n_gs)
        # gs_counter = render_pkg["gaussians_pixel_counter"]
        # T_cache[idx] = torch.nan_to_num(T / gs_counter, nan=0.0) # the case when counter = 0 , which means non-visible gaussians
        forward_vector = pair_cam.c2w[:3, 2]
        unit_forward_vector = forward_vector / torch.norm(forward_vector, dim=0)
        
        # import pdb; pdb.set_trace()
        
        direction_cache[idx] = render_pkg["visibility_filter"][..., None] * unit_forward_vector.to(opacity.device)

    direction_cache = direction_cache.permute(1, 0, 2) # (n_gs, n_cams, 3)

    pred_imgs = []
    for idx, cam in tqdm(enumerate(query_cameras), desc="query-cameras / get predicted image"):
        render_pkg = renderer(cam, gaussians, pipeline, background)
        pred_img = render_pkg["render"]
        pred_imgs.append(pred_img.clone().detach().cpu())
    
    for idx, cam in tqdm(enumerate(query_cameras), desc="query-cameras / get mask"):
        query_forward_vector = cam.c2w[:3, 2].to(opacity.device)
        unit_query_forward_vector = query_forward_vector / torch.norm(query_forward_vector, dim=0)
        chunk_size = 8192
        override_color = torch.zeros_like(direction_cache[:, 0, 0], device=direction_cache.device)
        for i in range(0, direction_cache.shape[0], chunk_size):
            chunk = direction_cache[i:i+chunk_size]
            dot_product = torch.sum(chunk * unit_query_forward_vector, dim=-1)
            override_color[i:i+chunk_size] = torch.max(dot_product, dim=1)[0]
        override_color = override_color.unsqueeze(1).repeat(1, 3)
        # print("[DEBUG] : override_color.min = ", override_color.min(), "override_color.max = ", override_color.max())
        render_pkg = renderer(cam, gaussians, pipeline, background, override_color=override_color)
        
        cos_sim_map = render_pkg["render"]
        
        if save_mode:
            pred_img = pred_imgs[idx]
            # T_render = T_render.detach().cpu()
            # mask = mask.detach().cpu()
            # masked_img = pred_img.clone()
            # masked_img[mask > 0] = 0.5
            
            # combined_image = torch.cat((pred_img, T_render, mask, masked_img), dim=-1)
            # print(pred_img.device, cos_sim_map.device)
            
            combined_image = torch.cat((pred_img, cos_sim_map.cpu()), dim=-1)
            save_path = os.path.join(render_path, "viewdirection_sim_map_" + cam.image_name + ".png")
            torchvision.utils.save_image(combined_image, save_path)

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
    return