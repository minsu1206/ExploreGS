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

_EPS = 1e-9
_INF = 1e10
_FEW = 5

# From FisherRF 
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

def render_fisher_set(model_path, 
                      split, 
                      loaded_iter, 
                      train_cameras, 
                      test_cameras, 
                      gaussians, 
                      renderer, 
                      pipeline,
                      mask_params,
                      background, 
                      save_mode=True, 
                      min_unc=-3, max_unc=6):
    """
    renderer : ModifiedGaussianRasterizer
    save_mode : default is True for rendering / set False when using diffusion inpainter prior
    """
    render_path = os.path.join(model_path, f"log-{loaded_iter:04d}", "uncertainty_fisher")
    if save_mode:
        os.makedirs(render_path, exist_ok=True)
    os.makedirs(os.path.join(render_path, "depth_heatmap"), exist_ok=True)
    os.makedirs(os.path.join(render_path, "depth"), exist_ok=True)
    params = list(capture(gaussians)[1:7])
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    xyz = params[0]
    
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i in mask_params.filter]
    
    optim = torch.optim.SGD(params, 0.)
    original_optim = gaussians.optimizer
    gaussians.optimizer = optim # This change GaussianModel's optimizer.
    device = params[0].device
    H_per_gaussian = torch.zeros(params[0].shape[0], device=params[0].device, dtype=params[0].dtype)
    
    pred_imgs = []
    uncertainty_maps = []
    aug_mask = []
    
    
    if mask_params.visibility_weight:
        visibility_cache = torch.zeros((len(train_cameras), len(gaussians._xyz))).to(gaussians._xyz.device)
        with torch.no_grad():
            for idx, view in enumerate(train_cameras):
                render_pkg = renderer(view, gaussians, pipeline, background)
                visibility = render_pkg["visibility_filter"]    
                visibility_cache[idx] = visibility
        visibility_weight = 1 - visibility_cache.sum(dim=0) / len(train_cameras)
        
    if not mask_params.fisher_current:
        
        if mask_params.skip_train and mask_params.skip_test:
            raise ValueError("skip_train and skip_test cannot be both True")
        
        if not mask_params.skip_train:
            for idx, view in enumerate(tqdm(train_cameras, desc="Rendering progress - FisherInfo - Train")):
                render_pkg = renderer(view, gaussians, pipeline, background)
                pred_img = render_pkg["render"]
                pred_img.backward(gradient=torch.ones_like(pred_img))
                pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
                # render_pkg = modified_render(view, gaussians, pipeline, background, override_color=torch.ones_like(params[1]))
                H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
                # render_pkg = modified_render(view, gaussians, pipeline, background, override_color=H_per_gaussian.detach())
                optim.zero_grad(set_to_none = True) 
                
            # CUDA Mem
            # print(f"[CUDA] : fisher_info : train-cameras bwd {idx} {(torch.cuda.memory_allocated()/1024**3):.3f}")
            # print(f"[CUDA] : fisher_info : train-cameras bwd {idx} {(torch.cuda.memory_reserved()/1024**3):.3f}")
        
        # # # # # # # # # # SKIP TEST # # # # # # # # # #
        if not mask_params.skip_test:
            for idx, view in enumerate(tqdm(test_cameras, desc="Rendering progress - FisherInfo - Test")):
                render_pkg = renderer(view, gaussians, pipeline, background)
                pred_img = render_pkg["render"]
                pred_img.backward(gradient=torch.ones_like(pred_img))
                pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
                H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
                optim.zero_grad(set_to_none = True)
            
            # CUDA Mem
            # print(f"[CUDA] : fisher_info : test-cameras bwd {idx} {(torch.cuda.memory_allocated()/1024**3):.3f}")
            # print(f"[CUDA] : fisher_info : test-cameras bwd {idx} {(torch.cuda.memory_reserved()/1024**3):.3f}")
                    
        # original
        hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3).clone()

        if mask_params.visibility_weight:
            hessian_color *= visibility_weight[..., None]
        
        with torch.no_grad():
            for idx, view in enumerate(tqdm(test_cameras, desc="Rendering on test set")):
                
                to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
                pts3d_homo = to_homo(xyz)
                pts3d_cam = pts3d_homo @ view.world_view_transform
                gaussian_depths = pts3d_cam[:, 2, None]

                cur_hessian_color = hessian_color * gaussian_depths.clamp(min=0) if not mask_params.nodepth else hessian_color

                pred_img, uncertainty_map, pixel_gaussian_counter = render_uncertainty(renderer, view, gaussians, pipeline, background, cur_hessian_color)

                intrinsic_aug_mask = process_intrinsic_aug_mask(view)
                if intrinsic_aug_mask is not None:
                    aug_mask.append(intrinsic_aug_mask)
                    os.makedirs(os.path.join(render_path, "aug_mask"), exist_ok=True)
                    torchvision.utils.save_image(intrinsic_aug_mask.clamp(0, 1), os.path.join(render_path, f"aug_mask/aug_mask_{view.image_name}.png"))
                    masked_img = pred_img * (1 - intrinsic_aug_mask[None].clamp(0, 1))
                    torchvision.utils.save_image(masked_img, os.path.join(render_path, f"aug_mask/masked_img_{view.image_name}.png"))
                    
                    if mask_params.replace_gt:
                        h_resize = view.image_height / view.fl_expand
                        w_resize = view.image_width / view.fl_expand
                        h_offset = int((h - h_resize) // 2)
                        w_offset = int((w - w_resize) // 2)
                        resized_replace_img = F.interpolate(view.replace_img, (int(h_resize), int(w_resize)), mode="bilinear")
                        pred_img[:,h_offset:h-h_offset, w_offset:w-w_offset] = resized_replace_img.to(dtype=pred_img.dtype, device=pred_img.device)

                pred_imgs.append(pred_img)
                
                heatmap = torch.log(uncertainty_map / (pixel_gaussian_counter + _EPS)).detach().cpu().clip(min_unc, max_unc)
                heatmap[pixel_gaussian_counter == 0] = max_unc
                uncertainty_maps.append(heatmap.to(dtype=uncertainty_map.dtype))
                # Process and save images
                if save_mode:
                    thresholds = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] # hardcoded
                    process_thresholding(heatmap, pred_img, view, render_path, thresholds)
                plt.clf()
                
                # CUDA Mem
                # print(f"[CUDA] : fisher_info : test-cameras render {idx} {(torch.cuda.memory_allocated()/1024**3):.3f}")
                # print(f"[CUDA] : fisher_info : test-cameras render {idx} {(torch.cuda.memory_reserved()/1024**3):.3f}")
    # # # # # # # # # # CURRENT # # # # # # # # # #
    else:
        target = test_cameras if split in ["test", "virtual"]  else train_cameras
        for idx, view in enumerate(tqdm(target, desc="Rendering on test set")):
            render_pkg = renderer(view, gaussians, pipeline, background)    
            pred_img = render_pkg["render"]
            depth = render_pkg["depth"]
            
            intrinsic_aug_mask = process_intrinsic_aug_mask(view)
            if intrinsic_aug_mask is not None:
                aug_mask.append(intrinsic_aug_mask)
                os.makedirs(os.path.join(render_path, "aug_mask"), exist_ok=True)
                torchvision.utils.save_image(intrinsic_aug_mask.clamp(0, 1), os.path.join(render_path, f"aug_mask/aug_mask_{view.image_name}.png"))
                masked_img = pred_img.detach().cpu() * (1 - intrinsic_aug_mask[None].clamp(0, 1))
                torchvision.utils.save_image(masked_img, os.path.join(render_path, f"aug_mask/masked_img_{view.image_name}.png"))
                
            # depth_uncertainty = (depth[1, :, :] - depth[0, :, :]).abs() / depth[1, :, :]
            # depth_heatmap = torch.log(depth_uncertainty + 1e-9).detach().cpu().clip(-2, 1)
            # sns.heatmap(depth_heatmap, square=True, vmin=-2, vmax=1)
            # plt.savefig(os.path.join(render_path, f"depth_heatmap/{view.image_name}.jpg"))
            # plt.clf()
            # depth_dist = depth[2, :, :]
            # print("[DEBUG] : depth_dist : ", depth_dist.max(), depth_dist.min(), depth_dist.mean(), depth_dist.std(), torch.log10(depth_dist + 1e-9).max(), torch.log10(depth_dist + 1e-9).min())
            # depth_heatmap = torch.log(depth_dist + 1e-9).detach().cpu().clip(-3, 3)
            # sns.heatmap(depth_heatmap, square=True, vmin=-3, vmax=6)
            # plt.savefig(os.path.join(render_path, f"depth_heatmap/{view.image_name}_dist.jpg"))
            # plt.clf()
            # print("[DEBUG] : depth : ", depth.shape, depth.max(), depth.min(), depth.mean(), depth.std())

            # median_depth = depth[1, :, :]
            # mean_depth = depth[0, :, :]
            # sns.heatmap(torch.log(median_depth.detach().cpu() + 1e-9), square=True, vmin=-5, vmax=5)
            # plt.savefig(os.path.join(render_path, f"depth/median_depth_{view.image_name}.png"))
            # plt.clf()
            # sns.heatmap(torch.log(mean_depth.detach().cpu() + 1e-9), square=True, vmin=-5, vmax=5)
            # plt.savefig(os.path.join(render_path, f"depth/mean_depth_{view.image_name}.png"))
            # plt.clf()
            
            # median_depth_color = apply_float_colormap(median_depth, colormap="magma")
            # mean_depth_color = apply_float_colormap(mean_depth, colormap="magma")
            
            # torchvision.utils.save_image(median_depth_color, os.path.join(render_path, f"depth/median_depth_{view.image_name}.png"))
            # torchvision.utils.save_image(mean_depth_color, os.path.join(render_path, f"depth/mean_depth_{view.image_name}.png"))
            
            pred_img.backward(gradient=torch.ones_like(pred_img))
            pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
            H_per_gaussian = sum(reduce(p.grad.detach(), "n ... -> n", "sum") for p in params)
            
            with torch.no_grad():
                hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3).clone()        
                
                if mask_params.visibility_weight:
                    hessian_color *= visibility_weight[..., None]

                to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
                pts3d_homo = to_homo(xyz)
                pts3d_cam = pts3d_homo @ view.world_view_transform
                gaussian_depths = pts3d_cam[:, 2, None]

                cur_hessian_color = hessian_color * gaussian_depths.clamp(min=0) if not mask_params.nodepth else hessian_color

                pred_img, uncertainty_map, pixel_gaussian_counter = render_uncertainty(renderer, view, gaussians, pipeline, background, cur_hessian_color)
                if mask_params.replace_gt:
                    print("[DEBUG] : replace_gt : ", view.image_name)
                    img_h = view.image_height
                    img_w = view.image_width
                    h_resize = int(img_h / view.fl_expand)
                    w_resize = int(img_w / view.fl_expand)
                    h_offset = int((img_h - h_resize) // 2)
                    w_offset = int((img_w - w_resize) // 2)
                    resized_replace_img = F.interpolate(view.replace_img[None], (int(h_resize), int(w_resize)), mode="bilinear")[0]
                    # print("[DEBUG] : replace_gt : ", h_offset, w_offset, h_resize, w_resize, pred_img.shape, resized_replace_img.shape,
                        #   img_h, img_w, pred_img[:,h_offset:h_resize+h_offset, w_offset:w_resize+w_offset].shape)
                    pred_img[:,h_offset:h_offset+h_resize, w_offset:w_offset+w_resize] = resized_replace_img.to(dtype=pred_img.dtype, device=pred_img.device)
            
                pred_imgs.append(pred_img)
                
                heatmap = torch.log(uncertainty_map / (pixel_gaussian_counter +1) + _EPS).detach().cpu().clip(min_unc, max_unc)
                heatmap[pixel_gaussian_counter == 0] = max_unc
                uncertainty_maps.append(heatmap.to(dtype=uncertainty_map.dtype))
                
                if save_mode:
                    thresholds = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] # hardcoded
                    process_thresholding(heatmap, pred_img, view, render_path, thresholds)
                plt.clf()
    
    render_pkg = {"pred_img":pred_imgs, "uncertainty_map":uncertainty_maps}
    if len(aug_mask) > 0:
        render_pkg["aug_mask"] = aug_mask
    gaussians.optimizer = original_optim
    return render_pkg
    
@torch.no_grad()
def render_uncertainty(renderer, view, gaussians, pipeline, background, hessian_color):
    # TODO
    render_pkg = renderer(view, gaussians, pipeline, background)
    pred_img = render_pkg["render"]
    pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
    
    render_pkg = renderer(view, gaussians, pipeline, background, override_color=hessian_color)

    uncertanity_map = reduce(render_pkg["render"], "c h w -> h w", "mean")

    return pred_img, uncertanity_map, pixel_gaussian_counter

# From nerfstudio/utils/colormap
def apply_float_colormap(image, colormap = "viridis"):
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    if colormap == "default":
        colormap = "turbo"

    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[image_long[..., 0]]


# Function to process and save heatmaps and masked images
def process_thresholding(heatmap, pred_img, view, eval_path, thresholds):
    # Save the initial heatmap visualization
    sns.heatmap(heatmap, square=True, vmin=-3, vmax=6)
    os.makedirs(os.path.join(eval_path, "bin"), exist_ok=True)
    os.makedirs(os.path.join(eval_path, "masked"), exist_ok=True)
    os.makedirs(os.path.join(eval_path, "concat"), exist_ok=True)
    os.makedirs(os.path.join(eval_path, "heatmap"), exist_ok=True)
    plt.savefig(os.path.join(eval_path, f"heatmap/heatmap_{view.image_name}.jpg"))
    plt.close()  # Close the plot to avoid overlapping with future plots
    pred_img = pred_img.cpu()
    heatmap = heatmap.cpu()
    
    for threshold in thresholds:
        # Create binary mask based on threshold
        binary = (heatmap.clone().detach() > threshold).float()
        
        # Apply binary mask to create the masked image
        masked_img = pred_img * (1 - binary[None])
        
        # Concatenate binary and masked image
        concatenated_img = torch.cat((binary.repeat(3, 1, 1), masked_img), dim=-1)  # Adjust dim based on the image size and shape
        
        # Save binary, masked, and concatenated images
        torchvision.utils.save_image(binary, os.path.join(eval_path, f"bin/bin_{view.image_name}_{threshold:.1f}.png"))
        torchvision.utils.save_image(masked_img, os.path.join(eval_path, f"masked/masked_{view.image_name}_{threshold:.1f}.png"))
        torchvision.utils.save_image(concatenated_img, os.path.join(eval_path, f"concat/concat_{view.image_name}_{threshold:.1f}.png"))

def process_intrinsic_aug_mask(cam):
    if cam.fl_expand > 1:
        h = cam.image_height
        w = cam.image_width
        # top left bottom right
        # h_offset = int((h - h * 1 / cam.fl_expand)/2)
        # w_offset = int((w - w * 1 / cam.fl_expand)/2)
        h_resize = int(h * 1 / cam.fl_expand)
        w_resize = int(w * 1 / cam.fl_expand)
        h_offset = int((h - h_resize) // 2)
        w_offset = int((w - w_resize) // 2)
        # original_region = [h_offset, h - h_offset, w_offset, w - w_offset]
        # make binary mask where inside original region is < 0, outside is > 0
        mask = torch.ones((h, w)) * 10
        # print("[DEBUG] : ", original_region, h, w, cam.fl_expand, h_offset, w_offset)
        # mask[original_region[0]:original_region[1], original_region[2]:original_region[3]] = -10
        mask[h_offset:h_offset+h_resize, w_offset:w_offset+w_resize] = -10
        return mask
    else:
        return None
    