import torch
import sys
import os

sys.path.append(f"{os.getcwd()}/Depth-Anything-V2")
print(sys.path)
# exit()
from depth_anything_v2.dpt import DepthAnythingV2
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage
import torch.nn.functional as F
import json
import numpy as np
import cv2

class DepthExtractor:
    
    def __init__(self, dataset, device='cuda'):
        
        # read depth_params.json
        if "explore" in dataset.source_path:
            depth_params_path = os.path.join(dataset.source_path, "depth_params.json")
        elif "scannet" in dataset.source_path:
            depth_params_path = os.path.join(dataset.source_path, "dslr/colmap/depth_params.json")
        else:
            depth_params_path = os.path.join(dataset.source_path, "model_train/sparse/depth_params.json")
        # depth_params_path = os.path.join(dataset.source_path, "model_train/sparse/depth_params.json")
        with open(depth_params_path, "r") as f:
            depth_params = json.load(f)
        all_scales = np.array([depth_params[key]["scale"] for key in depth_params])
        all_offsets = np.array([depth_params[key]["offset"] for key in depth_params])
        if (all_scales > 0).sum():
            med_scale = np.median(all_scales[all_scales > 0])
        else:
            med_scale = 0
        self.med_scale = med_scale
        self.med_offset = np.median(all_offsets)
        self.mean_offset = np.mean(all_offsets)
        
        # DepthAnythingV2
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        depth_model = DepthAnythingV2(**model_configs['vitl'])
        depth_model.load_state_dict(torch.load(f"{os.getcwd()}/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth", map_location='cpu'))
        depth_model = depth_model.to(device).eval()
        
        self.depth_model = depth_model
        
        input_size = 512
        
    @torch.no_grad()
    def extract_depth(self, pkg, scaling=True, save_mode=False):
        # image : torch tensor (C, H, W)
        # cam : Camera class instance
        image = pkg["image"].clone().detach() # do not modify original image
        
        original_h, original_w = image.shape[-2:]
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # image = self.transform(image)
        resize_w = int(original_w // 14 * 14)
        resize_h = int(original_h // 14 * 14)
        
        print("[DEBUG] : image.shape = ", image.shape)
        # apply normalize to image
        if len(image.shape) == 4:
            image = image.squeeze(0)
        
        image[0, ...] = (image[0, ...] - 0.485) / 0.229
        image[1, ...] = (image[1, ...] - 0.456) / 0.224
        image[2, ...] = (image[2, ...] - 0.406) / 0.225
        image = image.unsqueeze(0)
        
        image_resized = F.interpolate(image, (resize_h, resize_w), mode="bilinear", align_corners=True)
        depth = self.depth_model(image_resized.cuda())
        
        depth = F.interpolate(depth[:, None], (original_h, original_w), mode="bilinear", align_corners=True)[0, 0]

        # (H, W)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        if save_mode:
            depth_norm_vis = depth_norm.detach().cpu().numpy().astype(np.uint8)
            cam = pkg["camera"]
            os.makedirs("debugging", exist_ok=True)
            save_path = os.path.join("debugging", "depth_" + cam.image_name + ".png")
            cv2.imwrite(save_path, depth_norm_vis)
        
        depth_norm /= 2**16 

        scale_info = self.depth_scaling(depth_norm, pkg)
        # print("[DEBUG] : scale_info : ", scale_info)
        # scale_info = {"scale": self.med_scale, "offset": self.med_offset, "med_scale": self.med_scale}
        # print("[DEBUG] : scale_info : ", scale_info)

        return depth_norm, scale_info
        
    # for virtual camera
    # refer utils/make_depth_scale.py
    def depth_scaling(self, depth, pkg):
        cam = pkg["camera"]
        pixel_gs_counter = pkg["pixel_gs_counter"] # (H, W)
        gaussians = pkg["gaussians"]
        visibility_filter = pkg["visibility_filter"]
        
        image_width, image_height = cam.image_width, cam.image_height
        mask = visibility_filter.nonzero() # (M, 1)
        # visible_gaussians = gaussians[mask] # (M, ) # instead pts
        pts = gaussians.get_xyz # (N, 3)
        visible_pts = pts[mask].squeeze(1) # (M, 3)
        valid_xys = pixel_gs_counter.nonzero().cpu().numpy() # (N, 2)
        
        # pts_ = warp visible points (world coordinate) to camera coordinate
        w2c = cam.c2w.inverse().to(device=visible_pts.device, dtype=visible_pts.dtype)
        pts_ = torch.matmul(visible_pts, w2c[:3, :3].T) + w2c[:3, 3]
        
        invcolmapdepth = 1. / pts_[:, 2] 
        invmonodepthmap = depth
        # already divide by 2**16
        
        s = invmonodepthmap.shape[0] / image_height
        maps = (valid_xys * s).astype(np.float32)
        # valid = (
        #     (maps[..., 0] >= 0) * 
        #     (maps[..., 1] >= 0) * 
        #     (maps[..., 0] < image_width * s) * 
        #     (maps[..., 1] < image_height * s) * (invcolmapdepth > 0)) # omit last option

        valid = ((maps[..., 0] >= 0) * (maps[..., 1] >= 0) * (maps[..., 0] < image_width * s) * (maps[..., 1] < image_height * s))

        if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
            # maps = maps[valid, :]
            # invcolmapdepth = invcolmapdepth[valid]
            # invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
            # import pdb; pdb.set_trace()
            ## Median / dev
            t_colmap = invcolmapdepth.median()
            s_colmap = torch.abs(invcolmapdepth - t_colmap).mean()

            t_mono = invmonodepthmap.median()
            s_mono = torch.abs(invmonodepthmap - t_mono).mean()
            scale = s_colmap / s_mono
            offset = t_colmap - t_mono * scale
        else:
            scale = torch.tensor([0.0])
            offset = torch.tensor([0.0])
        return {"scale": float(scale.item()), "offset": float(offset.item()), "med_scale": self.med_scale}