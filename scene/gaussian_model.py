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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, fov2focal, getProjectionMatrix
from utils.general_utils import strip_symmetric, build_scaling_rotation
import copy
import einops
import json

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        def rotation_activation_sphere(rotation):   # FIXME: ??
            rotation = torch.nn.functional.normalize(rotation)
            c = rotation.shape[1]
            # slice and copy
            # print(rotation[:2])
            rotation = rotation[:, 0]
            sphere_rot = einops.repeat(rotation, "n -> n c ", c=c)
            # print(sphere_rot[:2])
            return sphere_rot
        
        self.rotation_activation_sphere = rotation_activation_sphere

    def __init__(self, 
                dataset:dict,
                opt:dict,
                mask_params:dict
                ):
        self.dataset = dataset
        self.opt = opt
        self.mask_params = mask_params
        self.active_sh_degree = 0
        self.max_sh_degree = dataset.sh_degree
        self.update_sh_interval = opt.update_sh_interval  # SH degree update interval
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.inv_depth = opt.inv_depth
        self.setup_functions()
        self.setup_masking(mask_params)
        self.appearance_model = "expmaps" if opt.appearance_model == "expmaps" else None
        
    # Deprecated >> Need to re-arranged
    def setup_masking(self, mask_params):
        # fisherRF style
        self.use_fisher = mask_params.use_fisher
        # visibility style
        self.use_visibility = mask_params.use_visibility
        self.use_visibility_mask = mask_params.use_visibility_mask
        self.training_args = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args, mask_params, train=True):
        # training_args = opt
        # print("[DEBUG] :restore ; model_args = ", len(model_args))
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        if train:
            self.training_setup(training_args, mask_params)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if train:
            self.optimizer.load_state_dict(opt_dict)
            # print optimizer learning rate
            # for param_group in self.optimizer.param_groups:
            #     print(f"[DEBUG] : gaussian_model.py : restore : {param_group['name']} param_group['lr'] = {param_group['lr']}")

    def slicing(self, mask):
        # TODO: apply mask into all variables like var[mask]
        return (
            self.active_sh_degree,
            self._xyz[mask],
            self._features_dc[mask],
            self._features_rest[mask],
            self._scaling[mask],
            self._rotation[mask],
            self._opacity[mask],
            self.max_radii2D,
            self.xyz_gradient_accum[mask],
            self.denom[mask],
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    @property
    def get_scaling(self):
        # To support isotropic sphere
        scaling = self._scaling if not self.dataset.isotropic else self._scaling[:, 0].unsqueeze(1).repeat(1, 3)
        # scaling = torch.ones_like(scaling)
        if self.dataset.isotropic:
            print(scaling.mean(), scaling.median())
            scaling = torch.ones_like(self._scaling) * 0.01 # for debugging
            return scaling
    
        # scaling = self._scaling[:, 0].unsqueeze(1).repeat(1, 3)
        # return self.scaling_activation(self._scaling) # original
        return self.scaling_activation(scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        
        if self.dataset.isotropic:
            rotation = torch.zeros((self._rotation.shape[0], 4), device="cuda")
            rotation = rotation[:, 0]
        else:
            rotation = self._rotation
        
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_exposure(self):
        return self._exposure
    
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            if image_name in self.exposure_mapping:
                return self._exposure[self.exposure_mapping[image_name]]
            else:
                print(f"Exposure for {image_name} not found in pretrained exposures.")
                return torch.eye(3, 4, device="cuda")
        else:
            if image_name not in self.pretrained_exposures:
                print(f"Exposure for {image_name} not found in pretrained exposures.")
                return torch.eye(3, 4, device="cuda")
            return self.pretrained_exposures[image_name]
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        else:
            print("[DEBUG] : gaussian_model.py : oneupSHdegree : already at max SH degree")

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, cam_infos : list = None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
 
        if self.appearance_model == "expmaps":
            if cam_infos is None:
                raise ValueError("cam_infos is required for expmaps appearance model", cam_infos)
            
            self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
            self.pretrained_exposures = None
            exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
            self._exposure = nn.Parameter(exposure.requires_grad_(True))
        
    def training_setup(self, training_args, mask_config):
        """
        training_args = opt
        mask_config = mask_params
        """
        self.training_args = training_args
        self.mask_config = mask_config
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_optimizer = None
        if self.appearance_model == "expmaps":
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
            self.exposure_scheduler_args = get_expon_lr_func(lr_init=training_args.exposure_lr_init,
                                                            lr_final=training_args.exposure_lr_final,
                                                            lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                            lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                            max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.appearance_model == "expmaps":
            if self.pretrained_exposures is None:
                for param_group in self.exposure_optimizer.param_groups:
                    param_group['lr'] = self.exposure_scheduler_args(iteration)
                    
        for param_group in self.optimizer.param_groups:
            # print("[DEBUG] : gaussian_model.py : update_learning_rate : param_group : ", param_group["name"], param_group["lr"])
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_sh_degree(self, iteration):
        if iteration % self.update_sh_interval == 0 and iteration > 0:
            self.oneupSHdegree()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        if self.appearance_model == "expmaps":
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"[INFO] : gaussian_model.py : load_ply : Pretrained exposures loaded.")
            else:
                print(f"[WARNING] : gaussian_model.py : load_ply : exposure.json not found in {os.path.dirname(path)}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        print(f"[INFO] : gaussian_model.py : load_ply -> load {len(self._xyz)} points")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # TODO: uncertainty based pruning ...?

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    # from DreamGaussian
    @torch.no_grad()
    def extract_fields(self, resolution=256, num_blocks=64, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        import kiui
        kiui.lo(occ, verbose=1)

        return occ

    def export_ply(self, ply_path):
        
        xyz = self.get_xyz # (N, 3)

        # complete this function
        # Ensure the path is correct and prepare to write to the file
        with open(ply_path, 'w') as ply_file:
            # Write the PLY header
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {xyz.shape[0]}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")
            ply_file.write("end_header\n")
            
            # Write the 3D points (x, y, z) for each Gaussian
            for point in xyz:
                ply_file.write(f"{point[0]} {point[1]} {point[2]} 0 0 0\n")

    # reference : https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/thirdparty/gaussian_splatting/scene/oursfull.py#L1153
    def add_gaussians(self, input_pkg, subsample=50):
        # subsample to avoid OOM
        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0
        
        threshold = 100
        camera = input_pkg["camera"]
        image = input_pkg["image"]
        visibility_filter = input_pkg["visibility_filter"]
        pixel_gs_counter = input_pkg["pixel_gs_counter"]
        # gaussians = input_pkg["gaussians"] # = self
        invdepth_render = input_pkg["invdepth"]
        avgdepth = input_pkg["avgdepth"]
        render_func = input_pkg["renderer"]

        device = self._xyz.device # CUDA
        invdepth_pred = camera.invdepthmap.squeeze().to(device)
        
        image = image.to(device)
        
        fx = fov2focal(camera.FoVx, camera.image_width)  # fovx -> focal length x
        fy = fov2focal(camera.FoVy, camera.image_height) # fovy -> focal length y
        proj_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=camera.FoVx, fovY=camera.FoVy).to(device=device, dtype=torch.float32) # (4, 4)
        extrinsic = camera.c2w.to(device=device, dtype=torch.float32) # camera to world
        
        # import pdb; pdb.set_trace()
        # 1. extract depth where pixel_gs_counter == 0 (empty region)
        empty_region_mask = (pixel_gs_counter < threshold).to(device=device)
        if empty_region_mask.sum() == 0:
            return False
        
        # empty_region_depth_render = invdepth_render * empty_region_mask 
        empty_region_depth_pred = invdepth_pred * empty_region_mask
        empty_region_img = image * empty_region_mask
        # 2. project points
        empty_region_pxs = empty_region_mask.nonzero()
        
        if subsample > 0:
            empty_region_pxs = empty_region_pxs[torch.randperm(empty_region_pxs.shape[0])[:subsample]]
        
        ndcv = pix2ndc(empty_region_pxs[:, 0], camera.image_height)[..., None] # (N, 1)
        ndcu = pix2ndc(empty_region_pxs[:, 1], camera.image_width)[..., None] # (N, 1)
        px_homo = torch.cat([ndcu, ndcv, torch.ones_like(ndcu), torch.ones_like(ndcv)], dim=-1) # (N, 4) , (y,x,1,1)
        # import pdb; pdb.set_trace()
        
        local_pts_uv = torch.matmul(proj_matrix.inverse(), px_homo.T).T # (N, 4)
        direction_local = local_pts_uv / local_pts_uv[:, 3:] # (N, 4)
        
        local_pts = direction_local * 1/invdepth_pred[empty_region_pxs[:, 0], empty_region_pxs[:, 1]][..., None]
        local_pts[:, -1] = 1.0
        
        world_pts_homo = local_pts @ extrinsic.T
        world_pts = world_pts_homo / world_pts_homo[:, 3:] # (N, 4)
        
        new_pts = world_pts[:, :3] / world_pts[:, 3:] # (N, 3)
        new_pts = new_pts.to(device=device)
        
        # camera_coords = normalized_coords * 1/invdepth_pred[empty_region_pxs[:, 0], empty_region_pxs[:, 1]][..., None]
        # # cam_coords_homo =  np.append(camera_coords, 1.0) # torch version
        # cam_coords_homo = torch.cat([camera_coords, torch.ones_like(camera_coords[:, 0][..., None])], dim=-1) # (N, 4)
        # world_coords_homo = (extrinsic.float() @ cam_coords_homo.T).T # (N, 4)
        # new_pts = world_coords_homo[:, :3] / world_coords_homo[:, 3:] # (N, 3)
        new_features_dc = empty_region_img[:, empty_region_pxs[:, 0], empty_region_pxs[:, 1]].T.unsqueeze(1) # (N, 1, 3) # get RGB color from image but only at empty_region_pxs
        new_features_rest = torch.zeros((len(new_pts), self._features_rest.shape[1], self._features_rest.shape[2]), device=device)
        new_opacity = inverse_sigmoid(0.9 * torch.ones_like(new_pts[:, 0]))[..., None] # (N, 1) # init near 1?
        new_rotation = torch.zeros((len(new_pts), 4), device=device)
        
        tmpxyz = torch.cat((new_pts, self._xyz), dim=0)
        dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
        dist2 = dist2[:new_pts.shape[0]]
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        new_scales = torch.clamp(scales, -10, 1.0)
        
        # 3. add projected points to gaussians
        self.densification_postfix(new_pts, new_features_dc, new_features_rest, new_opacity, new_scales, new_rotation)
        return True

def gaussian_3d_coeff(xyzs, covs, mdist=False):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    if mdist:
        return (power * -2 + 1e-10).clamp(min=1e-10).sqrt()

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

