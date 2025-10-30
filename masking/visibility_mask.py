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
def render_visibility_mask(model_path,
                           name,
                           loaded_iter, 
                           query_cameras, 
                           pair_cameras, 
                           gaussians, 
                           renderer, 
                           pipeline,
                           background, 
                           save_mode=False, 
                           threshold=0.5,
                           vis_cam_index:int=None):
    """
    ExtraNeRF style ; not same as visibility.py
    
    Case (1) --> query_cameras[0] (1 camera) : pair_cameras[0] (K cameras)
    query cameras : List[cam1, cam2, ...]
    pair cameras : List[[cam1, cam2, cam3, ..., camK], [cam1, cam2, cam3, ..., camK], ...]
    
    Case (2) --> pair_cameras = all training cameras
    query_cameras : List[cam1, cam2, ...]
    pair cameras : List[[cam1, cam2, ..., camN]]
    
    temporal hard-coded logging
    mask2 : original style / K = 5 , 2nd largest value // neark
    mask3 : isotropic at getting transmittance and at rendering / K = 5 , 2nd largest value
    mask4 : elliposid at getting transmittance and isotropic at rendering / K = 5 , 2nd largest value
    mask5 : aggregate from all training cameras but consider view counts also / K = -1, mean value // global
    """
    K = "2nd"
    type_ = "global" if len(pair_cameras) == 1 and isinstance(pair_cameras[0], list) else "neark"
    render_path = os.path.join(model_path, f"log-{loaded_iter:04d}", f"visibility_mask{type_}_thrs{threshold}")
    if K == "2nd":
        render_path += '_2nd_largest'
    elif k == -1:
        render_path += '_mean'
    
    
    if save_mode:
        os.makedirs(render_path, exist_ok=True)
        # print("[INFO] : render_visibility_mask : save at ", render_path)
    
    save_mode = False
    verbose = False
    if not verbose:
        print("[INFO] : render_visibility_mask : verbose mode is off ...")
    
    params = list(capture(gaussians)[1:7])
    params = [p.requires_grad_(False) for i, p in enumerate(params)]
    # (1) test cameras -> mark visible gaussians
    # (2) render at nearest train cameras and track them
    # implementation note : https://www.notion.so/Visibility-Mask-3DGS-1318db77e98780e78c40e2c5054a8145
    n_gs = len(gaussians.get_xyz)
    opacity = gaussians.get_opacity.detach().requires_grad_(False)
    
    pixel_gaussian_counters = []
    visibility_filters = []
    invdepths = []
    avgdepths = []
    
    if type_ == "global": # pair cameras : List[[cam1, cam2, ..., camN]]
        
        T_cache = torch.zeros((len(pair_cameras[0]), n_gs), device=opacity.device) # (K, #_GS) but K = len(all_train_cams)
        visible_counters = torch.zeros(n_gs, device=opacity.device)
        
        for idx, pair_cam in (tqdm(enumerate(pair_cameras[0]), desc="pair-cameras / Case(2)") if verbose else enumerate(pair_cameras[0])):
            render_pkg = renderer(pair_cam, gaussians, pipeline, background)
            T = render_pkg["T"] # (n_gs)
            gs_counter = render_pkg["gaussian_pixel_counter"]
            T_cache[idx] = torch.nan_to_num(T / gs_counter, nan=0.0) # the case when counter = 0 , which means non-visible gaussians
            visible_counters += render_pkg["visibility_filter"]
        
        # mean
        # T_cache = T_cache.sum(dim=0)
        # T_cache_mean = T_cache / visible_counters
        # T_cache_mean = repeat(T_cache_mean, "n -> n c", c=3)
        
        # top5
        selected_T, _ = torch.topk(T_cache, k=min(5, len(T_cache)), dim=0)
        selected_T = selected_T[1] # (N)
        T_cache_mean = repeat(selected_T, "n -> n c", c=3)
        
        pred_imgs = []
        for idx, cam in (tqdm(enumerate(query_cameras), desc="query-cameras / Case(2)") if verbose else enumerate(query_cameras)):
            render_pkg = renderer(cam, gaussians, pipeline, background)
            pred_img = render_pkg["render"]
            pred_imgs.append(pred_img.clone().detach().cpu())
        
        uncertainty_maps = []
        # binary_masks = []
        for idx, cam in (tqdm(enumerate(query_cameras), desc="query-cameras / Case(2)") if verbose else enumerate(query_cameras)):
            render_pkg = renderer(cam, gaussians, pipeline, background, override_color=T_cache_mean)
            T_render = render_pkg["render"]
            
            pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
            visibility_filter = render_pkg["visibility_filter"]
            pixel_gaussian_counters.append(pixel_gaussian_counter)
            visibility_filters.append(visibility_filter)
            invdepth = render_pkg["invdepth"]
            avgdepth = render_pkg["depth"]
            invdepths.append(invdepth)
            avgdepths.append(avgdepth)
            
            mask = (T_render.norm(dim=0) < threshold)
            uncertainty_maps.append(1 - T_render.norm(dim=0).clamp(0, 1))
            # binary_masks.append(mask)

            if save_mode:
                pred_img = pred_imgs[idx]
                T_render = T_render.detach().cpu()
                mask = mask.repeat(3, 1, 1).detach().cpu()
                masked_img = pred_img.clone()
                masked_img[mask > 0] = 0.5
                
                combined_image = torch.cat((pred_img, T_render, mask, masked_img), dim=-1)
                save_path = os.path.join(render_path, "visibility_mask_" + cam.image_name + ".png")
                torchvision.utils.save_image(combined_image, save_path)

    else:
        for idx, view in tqdm(enumerate(query_cameras), desc="query-cameras"):
            
            # get visibility mask
            render_pkg = renderer(view, gaussians, pipeline, background)
            pred_img = render_pkg["render"].clone().detach()
        
            # pred_img = render_pkg["render"]
            # pred_img.backward(gradient=torch.ones_like(pred_img))
            
            # get transmittance from pair cameras ; aggregation
            T_cache = torch.zeros((len(pair_cameras[idx]), n_gs), device=opacity.device) # (K, #_GS)
            C, H, W = pred_img.shape
            # T_per_pixel = torch.zeros((H, W, n_gs, 3))
            
            for j, pair_view in enumerate(pair_cameras[idx]):
                
                # 2nd try : forward pass
                render_pkg = renderer(pair_view, gaussians, pipeline, background)
                T = render_pkg["T"] # (n_gs)
                gs_counter = render_pkg["gaussian_pixel_counter"]
                T_cache[j] = torch.nan_to_num(T / gs_counter, nan=0.0) # the case when counter = 0 , which means non-visible gaussians

            # TODO: ExtraNeRF style -> simply select 2nd largest value
            selected_T, _ = torch.topk(T_cache, k=2, dim=0)
            selected_T = selected_T[1] # (N)
            selected_T = repeat(selected_T, "n -> n c", c=3)
            # print("[DEBUG] : selected_T : ", selected_T.device)
            
            # override color as selected_T and render it again
            # gaussians.isotropic = True
            render_pkg = renderer(view, gaussians, pipeline, background, override_color=selected_T)
            T_render = render_pkg["render"]
            
            mask = T_render < threshold
            
            if save_mode:
                pred_img = pred_img.detach().cpu()
                T_render = T_render.detach().cpu()
                mask = mask.detach().cpu()
                masked_img = pred_img.clone()
                masked_img[mask > 0] = 0.5
                
                combined_image = torch.cat((pred_img, T_render, mask, masked_img), dim=-1)
                save_path = os.path.join(render_path, "visibility_mask_" + view.image_name + ".png")
                torchvision.utils.save_image(combined_image, save_path)
                
                ref = torch.stack([view.original_image for view in pair_cameras[idx]])
                save_path = os.path.join(render_path, "visibility_mask_" + view.image_name + "_ref.png")
                torchvision.utils.save_image(ref, save_path)
                
        # break
    if vis_cam_index is not None:
        vis_cam = query_cameras[vis_cam_index]
        render_pkg = renderer(cam, gaussians, pipeline, background)
        T = render_pkg["T"]
        gs_counter = render_pkg["gaussian_pixel_counter"]
        T = torch.nan_to_num(T / gs_counter, nan=0.0)
        
        vis_gs = copy.deepcopy(gaussians)
        
        # print(vis_gs._features_dc.shape)
        # print(vis_gs._features_rest.shape)
        
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
    
    result_dict = {"pred_img":pred_imgs, "uncertainty_map":uncertainty_maps, 
                   "pixel_gaussian_counter":pixel_gaussian_counters, "visibility_filter": visibility_filters,
                   "invdepth": invdepths, "avgdepth": avgdepths}
    return result_dict

@torch.no_grad()
def render_visibility_mask_bg_filter(model_path,
                                     name,
                                    loaded_iter, 
                                    query_cameras, 
                                    pair_cameras, 
                                    gaussians, 
                                    renderer, 
                                    pipeline,
                                    background, 
                                    save_mode=False, 
                                    threshold=0.5,
                                    vis_cam_index:int=None):
    K = "2nd"
    type_ = "global" if len(pair_cameras) == 1 and isinstance(pair_cameras[0], list) else "neark"
    render_path = os.path.join(model_path, f"visibility_mask{type_}_thrs{threshold}")
    if K == "2nd":
        render_path += '_2nd_largest'
    elif k == -1:
        render_path += '_mean'
    
    
    # if save_mode:/
    os.makedirs(render_path, exist_ok=True)
        # print("[INFO] : render_visibility_mask : save at ", render_path)
    
    save_mode = False
    verbose = False
    if not verbose:
        print("[INFO] : render_visibility_mask : verbose mode is off ...")
    
    params = list(capture(gaussians)[1:7])
    params = [p.requires_grad_(False) for i, p in enumerate(params)]
    # (1) test cameras -> mark visible gaussians
    # (2) render at nearest train cameras and track them
    # implementation note : https://www.notion.so/Visibility-Mask-3DGS-1318db77e98780e78c40e2c5054a8145
    n_gs = len(gaussians.get_xyz)
    opacity = gaussians.get_opacity.detach().requires_grad_(False)
    
    pixel_gaussian_counters = []
    visibility_filters = []
    invdepths = []
    avgdepths = []
    
    if type_ == "global": # pair cameras : List[[cam1, cam2, ..., camN]]
        
        T_cache = torch.zeros((len(pair_cameras[0]), n_gs), device=opacity.device) # (K, #_GS) but K = len(all_train_cams)
        visible_counters = torch.zeros(n_gs, device=opacity.device)
        
        for idx, pair_cam in (tqdm(enumerate(pair_cameras[0]), desc="pair-cameras / Case(2)") if verbose else enumerate(pair_cameras[0])):
            render_pkg = renderer(pair_cam, gaussians, pipeline, background)
            T = render_pkg["T"] # (n_gs)
            gs_counter = render_pkg["gaussian_pixel_counter"]
            T_cache[idx] = torch.nan_to_num(T / gs_counter, nan=0.0) # the case when counter = 0 , which means non-visible gaussians
            visible_counters += render_pkg["visibility_filter"]
        
        # mean
        # T_cache = T_cache.sum(dim=0)
        # T_cache_mean = T_cache / visible_counters
        # T_cache_mean = repeat(T_cache_mean, "n -> n c", c=3)
        
        # top5
        selected_T, _ = torch.topk(T_cache, k=min(5, len(T_cache)), dim=0)
        selected_T = selected_T[1] # (N)
        T_cache_mean = repeat(selected_T, "n -> n c", c=3)
        
        pred_imgs = []
        for idx, cam in (tqdm(enumerate(query_cameras), desc="query-cameras / Case(2)") if verbose else enumerate(query_cameras)):
            render_pkg = renderer(cam, gaussians, pipeline, background)
            pred_img = render_pkg["render"]
            pred_imgs.append(pred_img.clone().detach().cpu())
        
        uncertainty_maps = []
        binary_masks = []
        for idx, cam in (tqdm(enumerate(query_cameras), desc="query-cameras / Case(2)") if verbose else enumerate(query_cameras)):
            
            bg_filter = torch.ones_like(background) * 10
            bg_filter_pkg = renderer(cam, gaussians, pipeline, bg_filter)
            bg_render = bg_filter_pkg["render"]
            bg_mask = bg_render.mean(dim=0) > 9.0 # (10 - 0.1) # fixed value here ?!
            # 1 : bg , 0 : fg
            
            render_pkg = renderer(cam, gaussians, pipeline, background, override_color=T_cache_mean)
            T_render = render_pkg["render"]
            
            
            pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
            visibility_filter = render_pkg["visibility_filter"]
            pixel_gaussian_counters.append(pixel_gaussian_counter)
            visibility_filters.append(visibility_filter)
            invdepth = render_pkg["invdepth"]
            avgdepth = render_pkg["depth"]
            invdepths.append(invdepth)
            avgdepths.append(avgdepth)
            
            T_render_norm = T_render.norm(dim=0)
            T_render_norm[bg_mask] = 0.0
            
            mask = (T_render_norm < threshold)
            uncertainty_map = 1 - T_render_norm.clamp(0, 1)
            uncertainty_maps.append(uncertainty_map)
            uncertainty_vis = uncertainty_map.detach().cpu()
            torchvision.utils.save_image(uncertainty_vis, os.path.join(render_path, f"{cam.image_name}_uncertainty.png"))
            
            binary_masks.append(mask)

    else:
        for idx, view in tqdm(enumerate(query_cameras), desc="query-cameras"):
            
            # get visibility mask
            render_pkg = renderer(view, gaussians, pipeline, background)
            pred_img = render_pkg["render"].clone().detach()
        
            # pred_img = render_pkg["render"]
            # pred_img.backward(gradient=torch.ones_like(pred_img))
            
            # get transmittance from pair cameras ; aggregation
            T_cache = torch.zeros((len(pair_cameras[idx]), n_gs), device=opacity.device) # (K, #_GS)
            C, H, W = pred_img.shape
            # T_per_pixel = torch.zeros((H, W, n_gs, 3))
            
            for j, pair_view in enumerate(pair_cameras[idx]):
                
                # 2nd try : forward pass
                render_pkg = renderer(pair_view, gaussians, pipeline, background)
                T = render_pkg["T"] # (n_gs)
                gs_counter = render_pkg["gaussian_pixel_counter"]
                T_cache[j] = torch.nan_to_num(T / gs_counter, nan=0.0) # the case when counter = 0 , which means non-visible gaussians

            # TODO: ExtraNeRF style -> simply select 2nd largest value
            selected_T, _ = torch.topk(T_cache, k=2, dim=0)
            selected_T = selected_T[1] # (N)
            selected_T = repeat(selected_T, "n -> n c", c=3)
            # print("[DEBUG] : selected_T : ", selected_T.device)
            
            # override color as selected_T and render it again
            # gaussians.isotropic = True
            render_pkg = renderer(view, gaussians, pipeline, background, override_color=selected_T)
            T_render = render_pkg["render"]
            
            mask = T_render < threshold
            
            if save_mode:
                pred_img = pred_img.detach().cpu()
                T_render = T_render.detach().cpu()
                mask = mask.detach().cpu()
                masked_img = pred_img.clone()
                masked_img[mask > 0] = 0.5
                
                combined_image = torch.cat((pred_img, T_render, mask, masked_img), dim=-1)
                save_path = os.path.join(render_path, "visibility_mask_" + view.image_name + ".png")
                torchvision.utils.save_image(combined_image, save_path)
                
                ref = torch.stack([view.original_image for view in pair_cameras[idx]])
                save_path = os.path.join(render_path, "visibility_mask_" + view.image_name + "_ref.png")
                torchvision.utils.save_image(ref, save_path)
                
        # break
    if vis_cam_index is not None:
        vis_cam = query_cameras[vis_cam_index]
        render_pkg = renderer(cam, gaussians, pipeline, background)
        T = render_pkg["T"]
        gs_counter = render_pkg["gaussian_pixel_counter"]
        T = torch.nan_to_num(T / gs_counter, nan=0.0)
        
        vis_gs = copy.deepcopy(gaussians)
        
        # print(vis_gs._features_dc.shape)
        # print(vis_gs._features_rest.shape)
        
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
    
    result_dict = {"pred_img":pred_imgs, "uncertainty_map":uncertainty_maps, 
                   "pixel_gaussian_counter":pixel_gaussian_counters, "visibility_filter": visibility_filters,
                   "invdepth": invdepths, "avgdepth": avgdepths, "binary_mask": binary_masks}
    return result_dict