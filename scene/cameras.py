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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from kornia import create_meshgrid
import cv2

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,
                 image_name, uid, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 use_view_direction=False, override_width=None, override_height=None, gt_alpha_mask=None, 
                 cost=0, extra_image=None,
                 invdepthmap=None, depth_params=None # from updated 3DGS
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R # c2w 
        self.T = T # w2c
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.use_view_direction = use_view_direction

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # self.extra_image = extra_image.clamp(0.0, 1.0).to(self.data_device) if extra_image is not None else None
        self.original_image = image.clamp(0.0, 1.0) if image is not None else None
        self.extra_image = extra_image.clamp(0.0, 1.0) if extra_image is not None else None
        
        # for n-crops
        
        # 256x256 rendering for diffusion model input
        if self.original_image is not None:
            self.image_width = self.original_image.shape[2] if override_width is None else override_width
            self.image_height = self.original_image.shape[1] if override_height is None else override_height
        else:
            if override_width is None or override_height is None:
                raise ValueError("original_image is None, but override_width or override_height is not provided")
            self.image_width = override_width
            self.image_height = override_height
    
        # self.gt_alpha_mask = gt_alpha_mask
        # # print("[DEBUG] : gt_alpha_mask = ", type(gt_alpha_mask), type(self.original_image))
        # if gt_alpha_mask is not None and self.original_image is not None:
        #     self.original_image *= gt_alpha_mask
        # else:
        #     self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width))
        #     self.original_image *= self.gt_alpha_mask # redudant?

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.original_image[0]) # [H, W]
            resolution = (self.image_width, self.image_height)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None])

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # refer : https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/thirdparty/gaussian_splatting/scene/cameras.py
        self.rayo = None
        self.rayd = None
        # self.c2w = self.world_view_transform.T.inverse().cpu()
        w2c = np.eye(4)
        w2c[:3, :3] = R.transpose()
        w2c[:3,  3] = T
        self.c2w = torch.from_numpy(np.linalg.inv(w2c)).float()
        
        if self.use_view_direction:
            project_inv = self.projection_matrix.T.inverse()
            cam2world = self.world_view_transform.T.inverse()
            
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False)[0]
            pixgrid = pixgrid.to(self.data_device)
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)
            ndc_cam = torch.cat((ndcx, ndcy, torch.ones_like(ndcy) * 1.0, torch.ones_like(ndcy)), 2)
            projected_cam = ndc_cam @ project_inv.T
            dir_in_local = projected_cam / projected_cam[:, :, 3:]
            direction = dir_in_local[:,:,:3] @ cam2world[:3,:3].T
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2,0,1)
            self.rayd = rays_d.permute(2, 0, 1)
            self.rays = torch.cat([self.rayo, self.rayd], dim=0)[None] # [1, 6, H, W]
            # print(f"[DEBUG] : self.rayd={self.rayd.shape} self.rayo={self.rayo.shape}") # [1, 3, H, W]
            # shape ?

        self.last_iter = 0
        self.virtual = False
        self.cost = cost # virtual camera property : How far from train camera
        self.fl_expand = 1.0 # focal length expand
        self.adjust_block_size = False
        self.replace_img = None
        self.ref_img_name = "" # only for virtual camera
        self.px_conf = None
        self.traj_id = None # which trajectory this camera belongs to
        self.traj_order = None # order of the camera in the trajectory
        self.lookat_depth = None # depth of the lookat point
        
    def __repr__(self):
        return f"Camera(ref_img_name={self.ref_img_name}, image_name={self.image_name}, virtual={self.virtual})"
    
    def set_extra_img(self, img):
        self.extra_image = img.clamp(0.0, 1.0).to(self.data_device)

    def set_depthmap(self, depth, depth_params):
        if type(depth) == torch.Tensor:
            depth = depth.cpu().numpy()
        
        invdepthmap = depth
        
        self.depth_mask = torch.ones_like(self.original_image[0]) # [H, W]
        resolution = (self.image_width, self.image_height)
        self.invdepthmap = cv2.resize(invdepthmap, resolution)
        self.invdepthmap[self.invdepthmap < 0] = 0
        self.depth_reliable = True

        if depth_params is not None:
            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                self.depth_reliable = False
                self.depth_mask *= 0
            
            if depth_params["scale"] > 0:
                self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

        if self.invdepthmap.ndim != 2:
            self.invdepthmap = self.invdepthmap[..., 0]
        self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class TrainVCam:
    """
    not used at rendering ; just used at diffusion pipeline
    """
    def __init__(self, original_image, R, T, 
                 FoVx, FoVy,
                 c2w=None,
                image_name="",
                image_width=256,
                image_height=256
                 ):
        self.original_image = original_image
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.R = R
        self.T = T
        if c2w is None:
            w2c = np.eye(4)
            w2c[:3, :3] = R.T
            w2c[:3,  3] = T
            c2w = np.linalg.inv(c2w)
        self.c2w = torch.from_numpy(c2w)
        self.image_name = image_name
        self.image_width = image_width
        self.image_height = image_height
        