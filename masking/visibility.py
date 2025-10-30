import torch
from itertools import combinations
from scene import GaussianModel
import copy
import einops
import torchvision
import os
import seaborn as sns
import matplotlib.pyplot as plt

@torch.no_grad()
def render_visibility(model_path, 
                      split, 
                      loaded_iter, 
                      train_cameras, 
                      test_cameras, 
                      gaussians, 
                      renderer, 
                      pipeline, 
                      background, 
                      save_mode=True,
                      nearest_k=-1,
                      v_cam_generator=None
                      ):
    """
    nearest_k : if -1, use all train cameras
    """
    render_path = os.path.join(model_path, f"log-{loaded_iter:04d}", "visibility")
    if save_mode:
        os.makedirs(render_path, exist_ok=True)
    
    visibility_cache = torch.zeros((len(train_cameras), len(gaussians._xyz))).to(gaussians._xyz.device)
    for idx, view in enumerate(train_cameras):
        render_dict = renderer(view, gaussians, pipeline, background)
        visibility = render_dict["visibility_filter"]
        visibility_cache[idx] = visibility
    
    # TODO
    visibility_sum = visibility_cache.sum(dim=0)
    # all_visibile = index set of [visibility_cache == len(train_cameras)]
    all_visible_mask = (visibility_sum == len(train_cameras)).nonzero(as_tuple=True)[0]
    
    visibility_prob = visibility_sum / len(train_cameras)
    
    override_color = einops.repeat(visibility_prob, "n -> n c", c=3)
    
    # strategy 2
    pred_imgs = []
    uncertainty_maps = [] 
    for idx, view in enumerate(test_cameras):
        render_pkg = renderer(view, gaussians, pipeline, background)
        pred_img = render_pkg["render"]
        render_vis_pkg = renderer(view, gaussians, pipeline, background, override_color=override_color)
        visibility_map = einops.reduce(render_vis_pkg["render"], "c h w -> h w", "mean")
        pred_imgs.append(pred_img)
        uncertainty_maps.append(1 - visibility_map)
        
    # print(len(pred_imgs), len(visibility_maps), save_mode)
    if save_mode:
        for pred_img, visibility_map, view in zip(pred_imgs, uncertainty_maps, test_cameras):
            print(os.path.join(render_path, f"{split}_render_{view.image_name}.png"))
            torchvision.utils.save_image(pred_img.detach(), os.path.join(render_path, f"{split}_render_{view.image_name}.png"))
            torchvision.utils.save_image(visibility_map.detach(), os.path.join(render_path, f"{split}_visibility_{view.image_name}.png"))
            # visibiliy heatmap boundary : [0 ~ 1]
            sns.heatmap(visibility_map.cpu(), vmin=0, vmax=1)
            plt.savefig(os.path.join(render_path, f"{split}_visibility_heatmap_{view.image_name}.png"))
            plt.clf()
            
            for threshold in [3/len(train_cameras), 0.01, 0.05 , 0.1, 0.2, 0.5, 0.8, 0.9]:
                binary_mask = (visibility_map < threshold).float()
                torchvision.utils.save_image(binary_mask, os.path.join(render_path, f"{split}_visibility_binary_{view.image_name}_{threshold:.1f}.png"))
            
    # return pred_imgs, visibility_maps
    render_pkg = {"pred_img":pred_imgs, "uncertainty_map":uncertainty_maps}
    return render_pkg

    # # strategy 1 -> fail ?
    # sliced_args = gaussians.slicing(all_visible_mask)
    # new_gaussian = GaussianModel(gaussians.dataset, gaussians.opt, gaussians.mask_params)
    
    # all_visible_gaussians = copy.deepcopy(new_gaussian)
    # all_visible_gaussians.restore(sliced_args, gaussians.opt, gaussians.mask_params, train=False)
    
    # if nearest_k > 1: # find nearest cameras 
    #     if v_cam_generator is not None:
    #         all_cam_trans = v_cam_generator.train_cameras_all_trans
    #     else:
    #         all_cam_trans = np.zeros((len(cameras), 3))
    #         for idx, cam in enumerate(train_cameras):
    #             all_cam_trans[idx] = cam.T
        
    #     train_camera_neark_idx = []
    #     for test_view in test_cameras:
    #         dist = np.linalg.norm(test_view.T - all_cam_trans)
    #         nearest_k_idx = np.argsort(dist)[:nearest_k]
    #         train_camera_neark_idx.append(nearest_k_idx)
        
    # else: # use all training cameras
    #     train_camera_neark_idx = [range(len(train_cameras)) for i in range(len(train_cameras))]
    
    # for test_cam, train_cam_idx_set in zip(test_cameras, train_camera_neark_idx):
    #     stereo_pairs = list(combinations(train_cam_idx_set, 2))
    #     for pair in stereo_pairs:
    #         stereo_visibility = ((visibility_cache[pair[0]] + visibility_cache[pair[1]]) == 2).nonzero(as_tuple=True)[0]