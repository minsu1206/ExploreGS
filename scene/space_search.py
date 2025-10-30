import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.spatial import ConvexHull
from utils.mesh_utils import render_set_rgbd, to_cam_open3d, tsdf_mesh, post_process_mesh, voxelize_classifiy_mesh_kdtree, export_voxel_grid_as_ply, make_grid_from_obb, export_mesh_w_value_as_ply
from utils.search_utils import *
from utils.camera_utils import rotation_matrix_x, rotation_matrix_y, create_cam2world_matrix, create_cam2world_keep_origin
import open3d as o3d
from utils.general_utils import inverse_sigmoid
from simple_knn._C import distCUDA2
import torchvision
from scene.cameras import Camera
import time
from tqdm import tqdm
import os
from einops import reduce, repeat
from collections import defaultdict
import heapq
from utils.graphics_utils import fov2focal
import torch.nn.functional as F
import math
import time
import gc
import glob

_EPS = 1e-9

class SpaceSearch:
    """
    Space search for camera trajectory generation
    """
    def __init__(self, scene, gaussians, cam_generator, dataset, opt, pipe, render_func, render_fisher, diff_params, mask_params, cam_params):
        
        self.scene = scene
        self.gaussians = copy.deepcopy(gaussians)
        self.cam_generator = cam_generator
        self.dataset = dataset
        self.scene_name = self.dataset.source_path.split("/")[-1]
        self.opt = opt
        self.pipe = pipe
        self.render_func = render_func
        self.render_fisher = render_fisher
        self.diff_params = diff_params
        self.mask_params = mask_params
        self.cam_params = cam_params
        self.device = "cuda"
        self.train_cameras = sorted(scene.getTrainCameras(), key=lambda x: x.image_name)
        self.train_cameras_pos = np.array([cam.c2w[:3, 3] for cam in self.train_cameras]) # [N, 3]
        self.coarse_voxel_resolution = cam_params.search_coarse_voxel_resolution
        self.voxel_resolution = cam_params.search_voxel_resolution
        self.reg_lambda = 1e-6 # for uncertainty
        # self.freespace_obb = True # compute fine-grained obb
        self.bbox_type = cam_params.search_bbox
        # for search v1
        self.topk = cam_params.search_topk
        self.init_step_ratio = 0.02
        self.aux_bbox_ratio = cam_params.search_aux_bbox_ratio
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        self.ax = ax
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.bg_color = torch.tensor(bg_color, device=self.device, dtype=torch.float32)

        # COMMON
        self.search_strategy = cam_params.search_search_strategy
        self.expand_strategy = cam_params.search_expand_strategy
        self.filter_params = cam_params.search_filter_params
        self.fisher_trick = cam_params.search_fisher_trick
        self.dir_angle = cam_params.search_dir_angle
        self.gs_proj = cam_params.search_gs_proj
        
        # Voxel classification
        self.mesh_dist_threshold = cam_params.search_mesh_dist_threshold
        self.use_frontier = cam_params.search_use_frontier
        
        # search strategy - freespace
        self.space_sampling_strategy = cam_params.search_space_sampling_strategy
        
        # search strategy - dfs - stop condition
        self.travel_stop = cam_params.search_travel_stop
        print("[DEBUG] : travel_stop = ", self.travel_stop)
        
        # action policy (1) : anchor set
        self.anchor_set = cam_params.search_anchor_set
        self.anchor_set_angle = cam_params.search_anchor_set_angle
        self.anchor_set_angle_orbit = cam_params.search_anchor_set_angle_orbit
        
        # action policy (2) : px level uncertainty map
        self.search_px_downsample = cam_params.search_px_downsample
        
        self.subsample = len(self.train_cameras) // cam_params.search_subsample if cam_params.search_subsample > 0 else cam_params.search_subsample
        self.early_guard = cam_params.search_early_guard
        self.bg_reject_ratio = cam_params.search_bg_reject_ratio
        self.gain_weight = cam_params.search_gain_weight
        self.travel_control = cam_params.search_travel_control

        self.mesh_leave_largest = cam_params.search_mesh_leave_largest
        self.occ_threshold = cam_params.search_occ_threshold
        self.orbit_threshold = cam_params.search_orbit_threshold
        self.reject_depth = cam_params.search_reject_depth
        self.t_cache = None
        
        # Subtle
        self.dbg_dir = f"{self.scene.model_path}/space_search"
        os.makedirs(self.dbg_dir, exist_ok=True)
        
        if self.search_strategy != "dfs":
            raise NotImplementedError(f"[ERROR] : search_strategy {self.search_strategy} is not supported")
            
        self.voxel_check = self.in_free_space
        
        self.fix_right_vec = cam_params.search_fix_right_vec
        self.fix_up_vec = cam_params.search_fix_up_vec # up vector sign
        
        if self.filter_params != [0,1,2,3,5]:
            omitted = []
            name = ["xyz", "rgb", "sh", "scale", "rotation", "opacity"]
            for idx in [0,1,2,3,4,5]:
                if idx in self.filter_params:
                    continue
                else:
                    omitted.append(name[idx])
        
        os.makedirs(self.dbg_dir, exist_ok=True)
        
        self._dbg_all = True # HARDCODED
        
        self.image_width = None
        self.image_height = None

        self.skip_rejection = cam_params.search_skip_rejection
        self.skip_freespace_rejection = cam_params.search_skip_freespace_rejection
        
        self.no_use_occ = cam_params.search_grid_no_use_occ

        self.reverse_top = cam_params.search_reverse_top
        self.init_mesh()

        if self.gain_weight == "visibility": # continuous value
            self.load_t_cache()
            print("[INFO] : t_cache.shape : {}".format(self.t_cache.shape))
        
    def build(self):
        
        print("[INFO] : search info : ", "\n",
              "scene_name : ", self.scene_name, "\n",
              "coarse_voxel_resolution : ", self.coarse_voxel_resolution, "\n",
              "voxel_resolution : ", self.voxel_resolution, "\n",
              "space_unit : ", self.space_unit, "\n",
              "depth_threshold : ", self.depth_threshold, "\n",
              "topk : ", self.topk, "\n",
              )
        if self.search_strategy == "dfs":
            trajectory = self.search_dfs()
            dataset = self.cam_generator.wrap_trajectories(trajectory, search=True)
        else:
            raise NotImplementedError(f"[ERROR] : search_strategy {self.search_strategy} is not supported")
        return dataset
    
    @torch.no_grad()
    def load_t_cache(self):
        print("[INFO] : load_t_cache")
        t_cache = torch.zeros((len(self.train_cameras), len(self.gaussians.get_xyz)), device=self.device)
        for idx, cam in enumerate(self.train_cameras):
            render_pkg = self.render_func(cam, self.gaussians, self.pipe, self.bg_color)
            T = render_pkg["T"]
            gs_counter = render_pkg["gaussian_pixel_counter"]
            t_cache[idx] = torch.nan_to_num(T / gs_counter, nan=0.0) # the case when counter = 0 , which means non-visible gaussians
        top5, _ = torch.topk(t_cache, k=min(5, len(t_cache)), dim=0)
        top2 = top5[1] # (N)
        self.t_cache = top2

    def init_mesh(self):
        # 1. extract mesh using TSDF
        rgbs, depths = render_set_rgbd(self.scene, self.gaussians, self.dataset, self.pipe, self.render_func)
        dummy = rgbs[0]
        self.image_width = dummy.shape[2]
        self.image_height = dummy.shape[1]
        print("[INFO] : image_width = ", self.image_width, "image_height = ", self.image_height)
        
        cameras = self.scene.getTrainCameras()
        mesh, cam_info = tsdf_mesh(rgbs, depths, cameras, self.gaussians, self.cam_params, return_cam_info=True)
        radius = cam_info["radius"]
        self.cam_generator.cam_radius = radius
        center = cam_info["center"]
        """ scene is bounded by radius * 2"""
        depth_min_median = torch.cat(depths).view(len(depths), -1).min(dim=1).values.median()
        depth_threshold = depth_min_median * 0.5
        self.depth_threshold = depth_threshold.cpu().item()
        o3d.io.write_triangle_mesh(f"{self.dbg_dir}/{self.scene_name}_mesh_raw.ply", mesh)
        mesh = post_process_mesh(mesh, cluster_to_keep=self.mesh_leave_largest)
        
        #mesh-cam
        mesh_w_cam = copy.deepcopy(mesh)
        # TODO: 2-2. extend bounding box to include training camera positions
        for cam_pos in self.train_cameras_pos:
            # make sphere with radius 0.001
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            # color = red
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(cam_pos)
            mesh_w_cam += sphere
        
        o3d.io.write_triangle_mesh(f"{self.dbg_dir}/{self.scene_name}_w_cam.ply", mesh_w_cam)

        # 2. compute bbox
        obb = mesh_w_cam.get_oriented_bounding_box(robust=False)
        obb_points = obb.get_box_points()
        # Create a point cloud from these corner points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obb_points)
        obb_mesh, _ = pcd.compute_convex_hull()         # Compute the convex hull, which will produce a valid triangular mesh
        obb_mesh.compute_vertex_normals()

        # Save to OBJ (or any other supported format)
        o3d.io.write_triangle_mesh(f"{self.dbg_dir}/{self.scene_name}_oriented_bounding_box.obj", obb_mesh)
        # 2-2. axis-aligned bbox
        # obb = mesh.get_axis_aligned_bounding_box()
        # 2-3. get minimal bounding box
        # obb = mesh.get_minimal_oriented_bounding_box(robust=False)
        
        # 2-*. simplify mesh --> bottleneck for accurate freespace ?
        # [WARNING] This might lead to memory issue if scene mesh is too large
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=min(50000, int(len(mesh.triangles) * 0.2)))
        mesh_w_cam = mesh_w_cam.simplify_quadric_decimation(target_number_of_triangles=min(50000, int(len(mesh_w_cam.triangles) * 0.2)))
        
        # 3. [1] mesh based free space / occupied space / unexplored space
        # print("[INFO] : voxelize_classifiy_mesh_kdtree")
        # voxel_centers_global, labels = voxelize_classifiy_mesh_kdtree(mesh, obb)
        # print("[INFO] : export_voxel_grid_as_ply")
        # export_voxel_grid_as_ply(voxel_centers_global, labels, "voxel_labels.ply")
        
        # 3. [2] GS Rasterizer based free space / occupied space / unexplored space
        voxel_centers_local = make_grid_from_obb(obb, resolution=self.coarse_voxel_resolution)
        extents = np.asarray(obb.extent)
        rotation = np.asarray(obb.R)
        center = np.asarray(obb.center)
        print("[DEBUG] : radius = ", radius)
        print("[DEBUG] : extents = ", extents)
        self.space_unit = float(np.linalg.norm(extents) / self.coarse_voxel_resolution)
        # self.voxel_size = extents / self.voxel_resolution
        self.voxel_size = self.space_unit
        
        # distance_threshold = radius * self.init_step_ratio
        voxel_centers_global = (rotation @ voxel_centers_local.T).T + center
        free_voxel_indices, occupied_voxel_indices, unexplored_voxel_indices, frontier_voxel_indices = self.occupancy_track(voxel_centers_global, mesh, distance=False, use_frontier=self.use_frontier)

        labels = np.full(len(voxel_centers_global), 'unknown', dtype=object)
        labels[occupied_voxel_indices] = 'occupied'
        labels[free_voxel_indices] = 'free'
        labels[unexplored_voxel_indices] = 'unknown'
        export_voxel_grid_as_ply(voxel_centers_global, labels, f"{self.dbg_dir}/{self.scene_name}_voxel_labels_occ_track_{self.coarse_voxel_resolution}.ply")
        print("[INFO] : export voxel grid as ply")
        self.clean_up()
        self.scene_mesh = mesh
        
        self.scene_obb = obb
        self.labels = labels
        self.search_space = voxel_centers_global
        # self.step = np.linalg.norm(self.voxel_size) # fixed small value for smoothness
        self.bbox_min = self.search_space.min(axis=0)
        
        if self.bbox_type == "sphere":
            raise NotImplementedError(f"[ERROR] : bbox_type {self.bbox_type} is not supported")
        elif self.bbox_type == "cam-bbox":
            raise NotImplementedError(f"[ERROR] : bbox_type {self.bbox_type} is not supported")
            # # use camera center and radius
            # bbox_center = center
            # bbox_side_length = radius * self.aux_bbox_ratio
            # # TODO: create bbox mesh
            # obb.center = bbox_center
            # obb.extent = bbox_side_length
            # voxel_centers_global_in_bbox = []
            # for pos in voxel_centers_global:
            #     if obb.contains(pos):
            #         voxel_centers_global_in_bbox.append(pos)
            # print("[DEBUG] : after bbox : ", len(voxel_centers_global), "->", len(voxel_centers_global_in_bbox))
            # voxel_centers_global = np.array(voxel_centers_global_in_bbox)
    
        elif self.bbox_type == "freespace-obb":
            free_space = voxel_centers_global[labels == 'free']
            # TODO: new mesh containing free_space (voxel cell vertex coordinates) and camera positions
            free_space_mesh = create_free_space_mesh(free_space, self.train_cameras_pos)
            obb = free_space_mesh.get_oriented_bounding_box(robust=False)
            
            if np.linalg.norm(obb.extent) > np.linalg.norm(self.scene_obb.extent):
                print("[DEBUG] : obb.extent > scene_obb.extent ; override as previous obb")
                obb = self.scene_obb
                
            obb_points = obb.get_box_points()
            # Create a point cloud from these corner points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obb_points)
            obb_mesh, _ = pcd.compute_convex_hull()         # Compute the convex hull, which will produce a valid triangular mesh
            obb_mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(f"{self.dbg_dir}/{self.scene_name}_freespace_obb.obj", obb_mesh)
            voxel_centers_local = make_grid_from_obb(obb, resolution=self.voxel_resolution)
            extents = np.asarray(obb.extent)
            self.voxel_size = float(np.linalg.norm(extents) / self.voxel_resolution)
            rotation = np.asarray(obb.R)
            center = np.asarray(obb.center)
            voxel_centers_global = (rotation @ voxel_centers_local.T).T + center
            self.search_space = voxel_centers_global
            free_voxel_indices, occupied_voxel_indices, unexplored_voxel_indices, frontier_voxel_indices = self.occupancy_track(voxel_centers_global, mesh, distance=True, use_frontier=self.use_frontier)
            labels = np.full(len(voxel_centers_global), 'unknown', dtype=object)
            labels[occupied_voxel_indices] = 'occupied'
            labels[free_voxel_indices] = 'free'
            labels[unexplored_voxel_indices] = 'unknown' # no use unknowned ?
            if self.use_frontier > 0:
                labels[frontier_voxel_indices] = 'frontier'
            self.labels = labels
            # self.space_unit = np.linalg.norm(extents) / self.coarse_voxel_resolution
            # self.voxel_size = extents / self.voxel_resolution
            self.step = np.linalg.norm(self.voxel_size) # fixed small value for smoothness
            self.bbox_min = self.search_space.min(axis=0)
            # labels[unexplored_voxel_indices] = 'free'
            export_voxel_grid_as_ply(voxel_centers_global, labels, f"{self.dbg_dir}/{self.scene_name}_freespace_obb_voxel_labels_{self.voxel_resolution}.ply", self.use_frontier)
            # TODO: Reduce memory usage
            self.clean_up()

        # [250221 issue] # move from #mesh-cam
        for cam_pos in self.train_cameras_pos:
            # make sphere with radius 0.001
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            # color = red
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(cam_pos)
            mesh += sphere
        
        o3d.io.write_triangle_mesh(f"{self.dbg_dir}/{self.scene_name}_w_cam.ply", mesh)

        self.cam_radius = radius
        # self.depth_threshold = radius * 0.25 # hmm ... 이거 맞나 # radius --> too flexible
        self.bg_filter = torch.tensor([10.0, 10.0, 10.0], device=self.device, dtype=torch.float32)
        plt.cla()
        plt.clf()
        # exit()
    
    def search_dfs(self, candidate_views=None):
        """
        depth first search
        """
        
        key_filter = self.filter_params
        params = self.gaussians.capture()[1:7]
        params = [p.requires_grad_(True) for i, p in enumerate(params) if i in key_filter]
        
        kwargs = {}

        if self.subsample > 0:
            anchor_cameras = self.train_cameras[::self.subsample]
        elif self.subsample == 0:
            anchor_cameras = self.train_cameras[0:1]
        else:
            anchor_cameras = self.train_cameras
        
        print("[DEBUG] : self.expand_strategy : ", self.expand_strategy)
        
        for strategy in self.expand_strategy: # [Room for multiple strategies]
            if "nbvs" in strategy:
                expand_func = self.expand_trajectory_nbvs_dfs
                H_train = self.get_H_train(params).cpu()
                I_train = torch.reciprocal(H_train + self.reg_lambda)
                kwargs["I_train"] = I_train
            elif "px" in strategy:
                expand_func = self.expand_trajectory_px_dfs
                if "bg" in strategy:
                    include_bg = True
                else:
                    include_bg = False
                kwargs["include_bg"] = include_bg
            
            elif "view-coverage" == strategy:
                print("[INFO] : view-coverage strategy ; initialize")
                
                expand_func = self.expand_trajectory_view_coverage_dfs
                
                n_gs = len(self.gaussians.get_xyz)
                up_vec = np.array([cam.c2w[:3, 1] for cam in self.train_cameras])
                # mean up vector
                mean_up_vec = np.mean(up_vec, axis=0)
                mean_up_vec = mean_up_vec / np.linalg.norm(mean_up_vec)
                
                # TODO:
                # create vectors : from center to sphere surface, azimuth = 30, elevation = 30
                # but consider mean up vector ; world coordinate up vector created should be aligned with mean up vector
                # apply same rotation to all vectors
                views = make_sphere_view_directions(self.dir_angle, self.dir_angle, np.array([0,0,0]), mean_up_vec)
                view_dirs = np.array(views)[:, :3, 2] # forward vector
                
                grid_view_dirs = np.zeros((n_gs, len(view_dirs))) # binary # TODO: expand binary to continuous

                for tcam in self.train_cameras:
                    with torch.no_grad():
                        render_pkg = self.render_func(tcam, self.gaussians, self.pipe, self.bg_color)
                        mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy() # (N_gs) # binary
                    tcam_forward = tcam.c2w[:3, 2].numpy()
                    # get angle between tcam_forward and view_dirs
                    angle_sim = view_dirs @ tcam_forward
                    # get index of view_dirs
                    idx = np.argmax(angle_sim)
                    grid_view_dirs[mask_visible, idx] = 1
                
                kwargs["grid_view_dirs"] = grid_view_dirs
                kwargs["view_dirs"] = view_dirs

            elif "view-coverage-grid" == strategy:
                
                expand_func = self.expand_trajectory_view_coverage_dfs
                
                xyz = self.gaussians.get_xyz[None].detach().requires_grad_(False) # (N, 3) tensor
                occupied_set = torch.from_numpy(self.search_space[self.labels == "occupied"]).to(dtype=torch.float32, device=xyz.device) # (M, 3)
                dist = []
                chunk = 512
                for i in range(0, len(occupied_set), chunk):
                    cur_set = occupied_set[i:i+chunk]
                    cur_dist = torch.cdist(cur_set[None], xyz).squeeze().min(dim=1).values
                    dist.append(cur_dist)
                dist = torch.cat(dist, dim=0)
                near_gs_mask = (dist < self.voxel_size * 0.5).cpu().numpy()
                near_gs_grid = occupied_set[near_gs_mask] # (K, 3)
                
                up_vec = np.array([cam.c2w[:3, 1] for cam in self.train_cameras])
                # mean up vector
                mean_up_vec = np.mean(up_vec, axis=0)
                mean_up_vec = mean_up_vec / np.linalg.norm(mean_up_vec)
                views = make_sphere_view_directions(self.dir_angle, self.dir_angle, np.array([0,0,0]), mean_up_vec)
                view_dirs = np.array(views)[:, :3, 2] # forward vector
                
                grid_view_dirs = np.zeros((len(near_gs_grid), len(view_dirs)))
                
                # init dummy GS from near_gs_grid
                grid_gs_xyz = near_gs_grid.float().to(xyz.device)
                grid_gs_features_dc = torch.zeros((len(grid_gs_xyz), 1, 3)).to(xyz.device)
                grid_gs_features_rest = torch.zeros((len(grid_gs_xyz), 15, 3)).to(xyz.device)
                grid_gs_opacity = inverse_sigmoid(0.005 * torch.ones_like(grid_gs_xyz[:, 0]))[..., None]
                tmpxyz = torch.cat([grid_gs_xyz, xyz[0]], dim=0)
                dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
                dist2 = dist2[:len(grid_gs_xyz)]
                grid_gs_scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
                grid_gs_rotation = torch.zeros((len(grid_gs_xyz), 4)).to(xyz.device)
                track_gs = copy.deepcopy(self.gaussians)
                track_gs.densification_postfix(grid_gs_xyz, grid_gs_features_dc, grid_gs_features_rest, grid_gs_opacity, grid_gs_scaling, grid_gs_rotation)
                mask = torch.ones((len(track_gs.get_xyz))).to(xyz.device)
                mask[:len(grid_gs_xyz)] = 0
                track_gs.prune_points(mask.bool())
                
                # only leave grid_gs_xyz
                
                for cam in self.train_cameras:
                    with torch.no_grad():
                        render_pkg = self.render_func(cam, track_gs, self.pipe, self.bg_color)
                        mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy() # (N_gs) # binary
                    cam_forward = cam.c2w[:3, 2].numpy()
                    # get angle between tcam_forward and view_dirs
                    angle_sim = view_dirs @ cam_forward
                    # get index of view_dirs
                    idx = np.argmax(angle_sim)
                    grid_view_dirs[mask_visible, idx] = 1
                
                kwargs["grid_gs"] = track_gs 
                kwargs["grid_view_dirs"] = grid_view_dirs.copy() # keep original
                kwargs["view_dirs"] = view_dirs

            # update grid_view_dirs
            # forward vector -> view direction
            elif "view-coverage-grid-refine" == strategy:
                expand_func = self.expand_trajectory_view_coverage_refine_dfs
                xyz = self.gaussians.get_xyz[None].detach().requires_grad_(False) # (N, 3) tensor
                if self.no_use_occ:
                    occupied_set = torch.from_numpy(self.search_space).to(dtype=torch.float32, device=xyz.device) # (M, 3)
                else:
                    occupied_set = torch.from_numpy(self.search_space[self.labels == "occupied"]).to(dtype=torch.float32, device=xyz.device) # (M, 3)
                dist = []
                chunk = 512
                for i in range(0, len(occupied_set), chunk):
                    cur_set = occupied_set[i:i+chunk]
                    cur_dist = torch.cdist(cur_set[None], xyz).squeeze().min(dim=1).values
                    dist.append(cur_dist)
                dist = torch.cat(dist, dim=0)
                near_gs_mask = (dist < self.voxel_size * 0.5).cpu().numpy()
                near_gs_grid = occupied_set[near_gs_mask] # (K, 3)
                
                up_vec = np.array([cam.c2w[:3, 1] for cam in self.train_cameras])
                # mean up vector
                mean_up_vec = np.mean(up_vec, axis=0)
                mean_up_vec = mean_up_vec / np.linalg.norm(mean_up_vec)
                views = make_sphere_view_directions(self.dir_angle, self.dir_angle, np.array([0,0,0]), mean_up_vec)
                view_dirs = np.array(views)[:, :3, 2] # forward vector # (M, 3)
                
                grid_view_dirs = np.zeros((len(near_gs_grid), len(view_dirs)))
                
                # init dummy GS from near_gs_grid
                grid_gs_xyz = near_gs_grid.float().to(xyz.device)
                grid_gs_features_dc = torch.zeros((len(grid_gs_xyz), 1, 3)).to(xyz.device)
                grid_gs_features_rest = torch.zeros((len(grid_gs_xyz), 15, 3)).to(xyz.device)
                grid_gs_opacity = inverse_sigmoid(0.005 * torch.ones_like(grid_gs_xyz[:, 0]))[..., None]
                tmpxyz = torch.cat([grid_gs_xyz, xyz[0]], dim=0)
                dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
                dist2 = dist2[:len(grid_gs_xyz)]
                grid_gs_scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
                grid_gs_rotation = torch.zeros((len(grid_gs_xyz), 4)).to(xyz.device)
                track_gs = copy.deepcopy(self.gaussians)
                track_gs.densification_postfix(grid_gs_xyz, grid_gs_features_dc, grid_gs_features_rest, grid_gs_opacity, grid_gs_scaling, grid_gs_rotation)
                mask = torch.ones((len(track_gs.get_xyz))).to(xyz.device)
                mask[:len(grid_gs_xyz)] = 0
                track_gs.prune_points(mask.bool())
                
                for tcam in self.train_cameras:
                    with torch.no_grad():
                        render_pkg = self.render_func(tcam, track_gs, self.pipe, self.bg_color)
                        mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy() # (N_gs) # binary
                    # compute view direction from cam to grid_gs_xyz
                    if self.gs_proj:
                        # 1. world2cam matrix into get_xyz result
                        # 2. rotate view_dirs by world2cam matrix
                        # 3. get angle between cam_view_dirs and view_dirs
                        # 4. get index of view_dirs
                        # 5. update grid_view_dirs
                        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
                        xyz = track_gs.get_xyz[mask_visible].detach()
                        pts3d_homo = to_homo(xyz)
                        pts3d_cam = pts3d_homo @ tcam.world_view_transform[:, :3] # (N_mask, 3)
                        cam_view_dirs = pts3d_cam - tcam.c2w[:3, 3].to(pts3d_cam.device) # (N_mask, 3)
                        cam_view_dirs = cam_view_dirs.cpu().numpy()
                        view_dirs = view_dirs @ tcam.world_view_transform[:3, :3].cpu().numpy().transpose() # (M, 3)
                    else:
                        cam_view_dirs = track_gs.get_xyz[mask_visible].detach().cpu().numpy() - tcam.c2w[:3, 3].numpy() # (N_gs, 3)
                    cam_view_dirs = cam_view_dirs / np.linalg.norm(cam_view_dirs, axis=1)[..., np.newaxis] # (N_gs, 3)
                    # get angle between cam_view_dirs and view_dirs
                    angle_sim = cam_view_dirs @ view_dirs.T # (N_gs, M)
                    # get index of view_dirs
                    idx = np.argmax(angle_sim, axis=1)
                    grid_view_dirs[mask_visible, idx] = 1

                kwargs["grid_gs"] = track_gs
                kwargs["grid_view_dirs"] = grid_view_dirs.copy() # keep original
                kwargs["view_dirs"] = view_dirs
                kwargs["use_grid"] = True
                print("[INFO] : view-coverage-grid-refine strategy ; initialize ; use_grid = {}".format(kwargs["use_grid"]))
                print("[INFO] : grid_view_dirs.sum() : {}".format(grid_view_dirs.sum()))
                print("[INFO] : grid_gs.get_xyz.shape : {}".format(track_gs.get_xyz.shape))
                
                
                t_cache = torch.zeros((len(self.train_cameras), len(track_gs.get_xyz)), device=self.device)
                for idx, cam in enumerate(self.train_cameras):
                    render_pkg = self.render_func(cam, track_gs, self.pipe, self.bg_color)
                    T = render_pkg["T"]
                    gs_counter = render_pkg["gaussian_pixel_counter"]
                    t_cache[idx] = torch.nan_to_num(T / gs_counter, nan=0.0) # the case when counter = 0 , which means non-visible gaussians
                top5, _ = torch.topk(t_cache, k=min(5, len(t_cache)), dim=0)
                top2 = top5[1] # (N)
                self.t_cache = top2
                print("[INFO] : self.t_cache.shape : {}".format(self.t_cache.shape))
                
            elif "view-coverage-refine" == strategy:
                expand_func = self.expand_trajectory_view_coverage_refine_dfs
                
                n_gs = len(self.gaussians.get_xyz)
                up_vec = np.array([cam.c2w[:3, 1] for cam in self.train_cameras])
                # mean up vector
                mean_up_vec = np.mean(up_vec, axis=0)
                mean_up_vec = mean_up_vec / np.linalg.norm(mean_up_vec)
                
                # TODO:
                # create vectors : from center to sphere surface, azimuth = 30, elevation = 30
                # but consider mean up vector ; world coordinate up vector created should be aligned with mean up vector
                # apply same rotation to all vectors
                views = make_sphere_view_directions(self.dir_angle, self.dir_angle, np.array([0,0,0]), mean_up_vec)
                view_dirs = np.array(views)[:, :3, 2] # forward vector # world space
                
                grid_view_dirs = np.zeros((n_gs, len(view_dirs))) # binary # TODO: expand binary to continuous

                for tcam in self.train_cameras:
                    with torch.no_grad():
                        render_pkg = self.render_func(tcam, self.gaussians, self.pipe, self.bg_color)
                        mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy() # (N_gs) # binary
                        
                    if self.gs_proj:
                        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
                        xyz = self.gaussians.get_xyz[mask_visible].detach()
                        pts3d_homo = to_homo(xyz)
                        pts3d_cam = pts3d_homo @ tcam.world_view_transform[:, :3] # (N_mask, 3)
                        cam_view_dirs = pts3d_cam - tcam.c2w[:3, 3].to(pts3d_cam.device) # (N_mask, 3)
                        cam_view_dirs = cam_view_dirs.cpu().numpy()
                        view_dirs = view_dirs @ tcam.world_view_transform[:3, :3].cpu().numpy().transpose() # (M, 3)
                    else:
                        cam_view_dirs = self.gaussians.get_xyz[mask_visible].detach().cpu().numpy() - tcam.c2w[:3, 3].numpy() # (N_gs, 3)
                    cam_view_dirs = cam_view_dirs / np.linalg.norm(cam_view_dirs, axis=1)[:, np.newaxis] # (N_gs, 3)
                    # get angle between cam_view_dirs and view_dirs
                    angle_sim = cam_view_dirs @ view_dirs.T # (N_gs, M)
                    # get index of view_dirs
                    idx = np.argmax(angle_sim, axis=1)
                    grid_view_dirs[mask_visible, idx] = 1
                
                kwargs["grid_gs"] = self.gaussians.get_xyz
                kwargs["grid_view_dirs"] = grid_view_dirs
                kwargs["view_dirs"] = view_dirs
                
                # if self.gs_proj: # debugging
                    
                #     grid_view_magnitude = grid_view_dirs.mean(axis=1) # (N_gs)
                #     grid_view_magnitude = grid_view_magnitude - grid_view_magnitude.min() # (N_gs)
                #     grid_view_magnitude = grid_view_magnitude / grid_view_magnitude.max() # (N_gs)
                #     # TODO:
                #     grid_xyz = self.gaussians.get_xyz.detach().cpu().numpy()    # (N_gs, 3)
                #     # Create an Open3D PointCloud
                #     pcd = o3d.geometry.PointCloud()
                #     pcd.points = o3d.utility.Vector3dVector(grid_xyz)

                #     # Map grid_view_magnitude to colors:
                #     #  - 0 => red (1,0,0)
                #     #  - 1 => blue (0,0,1)
                #     # You can pick your own interpolation; here's a simple linear blend:
                #     colors = np.zeros_like(grid_xyz)
                #     colors[:, 0] = 1 - grid_view_magnitude  # red channel decreases with magnitude
                #     colors[:, 2] = grid_view_magnitude      # blue channel increases with magnitude

                #     pcd.colors = o3d.utility.Vector3dVector(colors)

                #     # Save to a PLY file
                #     o3d.io.write_point_cloud("grid_view_magnitude.ply", pcd)
                #     print("[INFO] Saved grid_view_magnitude visualization to grid_view_magnitude.ply")
                #     exit()

            else:
                raise ValueError(f"[ERROR] : expand_strategy : {strategy} is not supported")

            optim = torch.optim.SGD(params, 0.)
            self.gaussians.optimizer = optim
        
            trajectories = []
            for anchor_cam in anchor_cameras:
                # TODO: add anchor set
                if "4o" in self.anchor_set:
                    anchor_cam = self.get_lookat_depth(anchor_cam)
                traj = Trajectory(anchor_cam)
                traj.path[0].traj_id = traj._traj_id
                traj.path[0].traj_order = len(traj.path)
                traj.path[0].ref_img_name = anchor_cam.image_name
                trajectories.append(traj)
            
            if self.early_guard == "force-free":
                # find closest free space and add_cam
                free_space = self.search_space[self.labels == "free"]
                print("[DEBUG] : early_guard = force-free")
                for traj in trajectories:
                    first_cam = traj.path[0]
                    first_cam_c2w = first_cam.c2w.numpy()
                    first_right_vec = first_cam_c2w[:3, 0]
                    first_up_vec = first_cam_c2w[:3, 1]
                    cam = traj.last
                    cam_pos = cam.c2w[:3, 3].numpy()
                    dist = np.linalg.norm(free_space - cam_pos, axis=1)
                    closest_idx = np.argmin(dist)
                    closest_free_space = free_space[closest_idx]
                    new_c2w = cam.c2w.numpy().copy()
                    new_c2w[:3, 3] = closest_free_space
                    if self.fix_right_vec:
                        new_c2w = modify_pose_rightvec(new_c2w, first_right_vec)
                    if self.fix_up_vec:
                        new_c2w = modify_pose_upvec_sign(new_c2w, first_up_vec)
                    new_w2c = np.linalg.inv(new_c2w)
                    vcam = self.create_vcam(cam, R=new_w2c[:3, :3].transpose(), T=new_w2c[:3, 3])
                    vcam = self.get_lookat_depth(vcam)
                    vcam.image_name = f"virtual_traj_{traj._traj_id}_1_forcefree"
                    vcam.traj_id = traj._traj_id
                    vcam.traj_order = 1
                    traj.add_cam(vcam)
            
            fx = fov2focal(anchor_cam.FoVx, anchor_cam.image_width)
            fy = fov2focal(anchor_cam.FoVy, anchor_cam.image_height)
            cx = anchor_cam.image_width / 2
            cy = anchor_cam.image_height / 2
            shared_K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            # initial_grid_view_dirs = kwargs["grid_view_dirs"]
            # # visualize
            # initial_gain = initial_grid_view_dirs.mean(axis=1) # [0~1] value range
            # # import pdb; pdb.set_trace()
            
            # initial_grid = kwargs["grid_gs"].detach().cpu().numpy()
            # # Create an Open3D PointCloud
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(initial_grid)

            # # Map initial_gain to colors:
            # #  - 0 => red (1,0,0)
            # # TODO: instead of red and blue, apply color map inferno
            # # colors = np.zeros_like(initial_grid)
            # # colors[:, 0] = 1 - initial_gain  # red channel decreases with magnitude
            # # colors[:, 2] = initial_gain      # blue channel increases with magnitude
            # colors = plt.get_cmap('inferno')(initial_gain)[:, :3]

            # # ERROR!!!!
            # pcd.colors = o3d.utility.Vector3dVector(colors)

            # # Save to a PLY file
            # o3d.io.write_point_cloud(f"{self.dbg_dir}/initial_grid.ply", pcd)
            
            # # load camera json -> update information gain
            # vcam_meta_data_dir = self.dbg_dir.replace("space_search", "vcam/metadata")
            # vcam_meta_data_dir = vcam_meta_data_dir.replace("final_vis", "v38_rot10_x2")
            # if not os.path.exists(vcam_meta_data_dir):
            #     print(f"[ERROR] : {vcam_meta_data_dir} does not exist")
            #     import pdb; pdb.set_trace()
            #     exit()
            # vcam_json_list = glob.glob(os.path.join(vcam_meta_data_dir, "*.json"))
            # print(f"[INFO] : Found {len(vcam_json_list)} vcam metadata")
            # # # import pdb; pdb.set_trace()
            
            # from utils.virtual_cam_utils import load_camera_metadata
            
            # for vcam_json_path in vcam_json_list:
            #     vcam = load_camera_metadata(vcam_json_path)
                
            #     with torch.no_grad():
            #         render_pkg = self.render_func(vcam, self.gaussians, self.pipe, self.bg_color)
            #         mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy() # (N_gs) # binary
            #     cam_view_dirs = self.gaussians.get_xyz[mask_visible].detach().cpu().numpy() - vcam.c2w[:3, 3].numpy() # (N_gs, 3)
            #     cam_view_dirs = cam_view_dirs / np.linalg.norm(cam_view_dirs, axis=1)[:, np.newaxis] # (N_gs, 3)
            #     # get angle between cam_view_dirs and view_dirs
            #     angle_sim = cam_view_dirs @ view_dirs.T # (N_gs, M)
            #     # get index of view_dirs
            #     idx = np.argmax(angle_sim, axis=1)
                
            #     new_grid_view_dirs = grid_view_dirs.copy()
            #     new_grid_view_dirs[mask_visible, idx] = 1
            #     gain = new_grid_view_dirs.sum() - grid_view_dirs.sum()
            #     print(f"[INFO] : vcam {vcam_json_path} : {gain}")
                
            #     grid_view_dirs = new_grid_view_dirs
                
            # final_gain = grid_view_dirs.mean(axis=1) # [0~1] value range
            # # visualize
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(initial_grid)
            # colors = plt.get_cmap('inferno')(final_gain)[:, :3]
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(f"{self.dbg_dir}/final_grid.ply", pcd)
            # print(f"[INFO] : Saved final grid visualization to {self.dbg_dir}/final_grid.ply")
            # # exit()
            
            terminated = []
            for traj in trajectories:
                stack = [traj]
                iter_ = 0
                while stack:
                    print(f"[DEBUG] : strategy : {strategy}, iter_ : {iter_}, len(stack) : {len(stack)}")
                    print("[DEBUG] : stack check : ", [len(traj) for traj in stack])
                    traj = stack.pop()
                    print("[DEBUG] : traj : ", traj.path, traj._traj_id)
                    if traj.terminate:
                        if self.travel_stop_by_path(traj):
                            print("[DEBUG] : terminate : ! ", len(traj), traj.root, traj._traj_id)
                            
                            terminated.append(traj)
                            # remove another paths with same root
                            # optional
                            new_stack = []
                            if self.travel_control == "greedy":    
                                for traj_ in stack:
                                    if traj_.root != traj.root:
                                        new_stack.append(traj_)
                                    else:
                                        print("[DEBUG] : greedy - remove : ", traj_.root)
                            elif self.travel_control == "all":
                                for traj_ in stack:
                                    new_stack.append(traj_)
                            elif self.travel_control == "drop-half":
                                for traj_ in stack:
                                    if traj_.root == traj.root:
                                        if len(traj_) < int(len(traj) / 2):
                                            new_stack.append(traj_)
                                        else:
                                            print("[DEBUG] : drop-half - remove : ", traj_.root)
                                    else:
                                        print("[DEBUG] : drop-half - remove : ", traj_.root)                                            
                            else:
                                raise ValueError(f"[ERROR] : travel_control : {self.travel_control} is not supported")
                            stack = new_stack
                            continue
                        else:
                            continue

                    new_trajs = expand_func(traj, shared_K, params, **kwargs)
                    if len(new_trajs) > 0:
                        for child_traj in new_trajs:
                            print("[DEBUG] : child_traj : ", child_traj.root, child_traj._traj_id)
                            stack.append(child_traj)
                    iter_ += 1

                if "view-coverage-refine" == strategy or "view-coverage-grid-refine" == strategy:
                    prev_grid_sum = kwargs["grid_view_dirs"].sum()
                    kwargs["grid_view_dirs"] = traj.grid_view_dirs
                    new_grid_sum = kwargs["grid_view_dirs"].sum()
                    print(f"[INFO] : update grid_view_dirs {prev_grid_sum} >> {new_grid_sum}")
                
        self.trajectories = terminated
        
        return self.trajectories
    
    # Deprecated ; need to be refactored
    def expand_trajectory_px_dfs(self, traj, shared_K, params, min_unc=-3.0, max_unc=6.0, include_bg=False):
        """
        Unified function for DFS-based trajectory expansion
        - Includes camera rotation
        - Operates at pixel level with optional background consideration
        """
        self.clean_up()
        cam = traj.last
        c2w = cam.c2w.float()
        xyz = params[0]
        
        if self.travel_stop_by_path(traj):
            print("[DEBUG] : dfs terminate : ", len(traj))
            traj.terminate = True
            return [traj]
        
        # if traj.check_stuck():
        #     print("[DEBUG] : dfs terminate : stuck")
        #     traj.terminate = True
        #     return [traj]
        
        print("[DEBUG] : cam : ", cam.image_name)
        traj_length = len(traj)
        save_name = f"virtual_traj_{traj._traj_id}_{len(traj)}"
        
        render_fisher_pkg = self.render_fisher(cam, self.gaussians, self.pipe, self.bg_color)
        pred_img = render_fisher_pkg["render"]
        # Uncomment If you want to track rejected viewpoints
        # torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{save_name}_fisher.png")
        pred_img.backward(gradient=torch.ones_like(pred_img))
        
        H_per_gs = sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
        H_per_gs_color = repeat(H_per_gs.detach(), "n -> n c", c=3).clone()
        
        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
        pts3d_homo = to_homo(xyz)
        pts3d_cam = pts3d_homo @ cam.world_view_transform
        gaussian_depths = pts3d_cam[:, 2, None]
        
        H_per_gs_color *= gaussian_depths.clamp(min=0)
        
        for p in params:
            p.grad = None
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        render_fisher_pkg2 = self.render_fisher(cam, self.gaussians, self.pipe, self.bg_color, override_color=H_per_gs_color)
        uncertainty_map = reduce(render_fisher_pkg2["render"], "c h w -> h w", "mean")
        pixel_gaussian_counter = render_fisher_pkg2["pixel_gaussian_counter"]
        
        for p in params:
            p.grad = None
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            render_pkg = self.render_func(cam, self.gaussians, self.pipe, self.bg_filter, skip_clamp=True)
        _, bg_mask = self.reject_cam_by_bg(render_pkg)
        
        uncertainty_map = torch.log(uncertainty_map / (pixel_gaussian_counter + 1) + _EPS).detach().cpu().clip(min_unc, max_unc)
        uncertainty_map[pixel_gaussian_counter == 0] = max_unc if include_bg else min_unc
        uncertainty_map[bg_mask] = torch.exp(torch.tensor([max_unc if include_bg else min_unc])).to(uncertainty_map.device)
        
        sns.heatmap(uncertainty_map.numpy(), square=True, vmin=min_unc, vmax=max_unc)
        plt.savefig(f"{self.dbg_dir}/{save_name}_unc_heatmap.jpg")
        plt.close()
        
        orig_h, orig_w = uncertainty_map.shape
        
        if self.search_px_downsample < 0:
            topk_values, topk_indices = torch.topk(uncertainty_map.view(-1), k=self.topk)
            patch_coords = torch.stack([torch.div(topk_indices, orig_w, rounding_mode='floor'), topk_indices % orig_w], dim=-1)
        else:
            down_size = self.search_px_downsample
            downsampled_uncertainty = F.interpolate(uncertainty_map[None, None], size=(down_size, down_size), mode="bilinear", align_corners=False)[0, 0]
            topk_values, topk_indices = torch.topk(downsampled_uncertainty.view(-1), k=1)
            patch_coords = torch.stack([torch.div(topk_indices, down_size, rounding_mode='floor'), topk_indices % down_size], dim=-1)
        
        dir_vecs = compute_viewing_direction(shared_K, c2w, patch_coords, (orig_h, orig_w))
        expansions = []
        
        for i, pos in enumerate(patch_coords):
            y, x = pos
            dir_vec = dir_vecs[i]
            new_position = c2w[:3, 3] + self.step * dir_vec
            
            current_forward = c2w[:3, 2]
            axis = np.cross(current_forward, dir_vec)
            axis_norm = np.linalg.norm(axis + 1e-15)
            
            if axis_norm < 1e-6:
                new_rotation = c2w[:3, :3]
            else:
                axis /= axis_norm
                angle = np.arccos(np.dot(current_forward, dir_vec)) / 10
                R_delta = torch.from_numpy(axis_angle_rotation(axis, angle)).to(device=c2w.device, dtype=c2w.dtype)
                new_rotation = R_delta @ c2w
            
            c2w_new = c2w.clone()
            c2w_new[:3, 3] = new_position
            c2w_new[:3, :3] = new_rotation[:3, :3]
            w2c = np.linalg.inv(c2w_new.numpy())
            
            v_cam = self.create_vcam(cam, R=w2c[:3, :3].transpose(), T=w2c[:3, 3])
            v_cam.image_name = f"{save_name}.png"
            with torch.no_grad():
                render_pkg = self.render_func(v_cam, self.gaussians, self.pipe, self.bg_color, skip_clamp=True)
            pred_img = render_pkg["render"]
            lists = glob.glob(f"{self.dbg_dir}/{save_name}*")
            count = len(lists)
            
            with torch.no_grad():
                render_pkg = self.render_func(v_cam, self.gaussians, self.pipe, self.bg_filter, skip_clamp=True)
            
            reject_bg, bg_mask = self.reject_cam_by_bg(render_pkg)
            
            # Uncomment If you want to track rejected viewpoints
            # if reject_bg:
            #     torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{save_name}_reject_bg.png")
            #     continue
            
            reject_depth, depth_mask = self.reject_cam_by_depth(render_pkg, bg_mask)
            if reject_depth:
                new_traj = copy.deepcopy(traj)
                v_cam.traj_id = new_traj._traj_id
                v_cam.traj_order = len(new_traj)
                new_traj.add_cam(v_cam)
                new_traj.terminate = True
                # Uncomment If you want to track rejected viewpoints
                # torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{save_name}_reject_depth.png")
                expansions.append(new_traj)
                continue
            
            is_free = self.voxel_check(new_position.numpy(), self.search_space, self.labels, radius=self.step * 1.5)
            if type(self.early_guard) == int:
                is_free = True if self.early_guard > len(traj) else is_free
            # Uncomment If you want to track rejected viewpoints
            # torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{save_name}_free_{is_free}.png")
            if is_free:
                new_traj = copy.deepcopy(traj)
                v_cam.traj_id = new_traj._traj_id
                v_cam.traj_order = len(new_traj)
                new_traj.add_cam(v_cam)
                expansions.append(new_traj)
        
        if expansions: # backward
            dir_vec = c2w[:3, 2]
            new_position = c2w[:3, 3] - self.step * dir_vec
            c2w_new = c2w.clone()
            c2w_new[:3, 3] = new_position
            c2w_new[:3, :3] = c2w[:3, :3]
            w2c_new = np.linalg.inv(c2w_new.numpy())
            v_cam = self.create_vcam(cam, R=w2c_new[:3, :3].transpose(), T=w2c_new[:3, 3])
            v_cam.image_name = f"{save_name}.png"
            new_traj = copy.deepcopy(traj)
            v_cam.traj_id = new_traj._traj_id
            v_cam.traj_order = len(new_traj)
            new_traj.add_cam(v_cam)
            expansions.append(new_traj)
        
        return expansions

    def expand_trajectory_nbvs_dfs(self, traj, shared_K, params, I_train, min_unc=-3.0, max_unc=6.0):
        self.clean_up()
        
        expansions = []
        
        cam = traj.last
        c2w = cam.c2w.float()
        c2w_np = c2w.numpy()

        first_cam = traj.path[0]
        first_cam_c2w = first_cam.c2w.numpy()
        first_right_vec = first_cam_c2w[:3, 0]

        if self.travel_stop_by_path(traj):
            print("[DEBUG] : dfs terminate : ", len(traj))
            traj.terminate = True
            return [traj]
        
        # if traj.check_stuck():
        #     print("[DEBUG] : dfs terminate : stuck")
        #     traj.terminate = True
        #     return [traj]
        
        # TODO: replace this as anchor set
        candidate_c2ws, names = self.action_anchor_set(c2w_np,
                                                       depth=cam.lookat_depth if "4o" in self.anchor_set else None)
        candidate_cams = []
        valid_names = []
        
        # make candidates
        for idx, (candidate_c2w, name) in enumerate(zip(candidate_c2ws, names)):
            
            name_rejected = False
            if len(traj.directions) > 0:
                name_rejected = self.camera_name_rejection(name, traj.directions[-1])
            
            if name_rejected:
                continue
            
            candidate_w2c = np.linalg.inv(candidate_c2w)
            v_cam = self.create_vcam(cam, R=candidate_w2c[:3, :3].transpose(), T=candidate_w2c[:3, 3])
            v_cam.image_name = f"virtual_traj_{traj._traj_id}_{len(traj)}_{name}"
            rejected = self.camera_rejection(v_cam)
            if rejected:
                continue
            else:
                candidate_cams.append(v_cam)
                valid_names.append(name)
        
        # invalid case
        if len(candidate_cams) == 0:
            # debugging session
            print("[DEBUG] : reject : no valid candidate cameras")
            self.debug_invalid_cam(cam)
            return []

        sorted_idx = self.policy_nbvs(params, candidate_cams, I_train)
        
        for p in params:
            p.grad = None
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        for i, idx in enumerate(sorted_idx[:self.topk]):
            selected_cam = candidate_cams[idx]
            
            with torch.no_grad():
                render_pkg = self.render_func(selected_cam, self.gaussians, self.pipe, self.bg_color)
            pred_img = render_pkg["render"]
            if "4o" in self.anchor_set:
                selected_cam = self.get_lookat_depth(selected_cam, render_pkg)
            # Uncomment If you want to track rejected viewpoints
            # torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{selected_cam.image_name}_nbvs_top{i}.png")
            
            new_traj = copy.deepcopy(traj)
            selected_cam.traj_id = new_traj._traj_id
            selected_cam.traj_order = len(new_traj)
            new_traj.add_cam(selected_cam)
            expansions.append(new_traj)
            
        return expansions[::-1] # best view comes last -> pop first !

    # TODO: merge with expand_trajectory_nbvs_dfs
    def expand_trajectory_view_coverage_dfs(self, traj, _, params, grid_view_dirs, view_dirs, grid_gs=None):
        """
        """
        self.clean_up()
        
        expansions = []
        
        cam = traj.last
        c2w = cam.c2w.float()
        xyz = params[0]

        if self.travel_stop_by_path(traj):
            print("[DEBUG] : dfs terminate : ", len(traj))
            traj.terminate = True
            return [traj]
        
        # if traj.check_stuck():
        #     print("[DEBUG] : dfs terminate : stuck")
        #     traj.terminate = True
        #     return [traj]
        
        c2w_np = c2w.numpy()
        candidate_c2ws, names = self.action_anchor_set(c2w_np, 
                                                       depth=cam.lookat_depth if "4o" in self.anchor_set else None)
        
        candidate_cams = []
        valid_names = []
        
        for idx, (candidate_c2w, name) in enumerate(zip(candidate_c2ws, names)):
            
            name_rejected = False
            if len(traj.directions) > 0:
                name_rejected = self.camera_name_rejection(name, traj.directions[-1])
            
            if name_rejected:
                continue
            
            candidate_w2c = np.linalg.inv(candidate_c2w)
            v_cam = self.create_vcam(cam, R=candidate_w2c[:3, :3].transpose(), T=candidate_w2c[:3, 3])
            v_cam.image_name = f"virtual_traj_{traj._traj_id}_{len(traj)}_{name}"
            rejected = self.camera_rejection(v_cam)
            # print(f"[DEBUG] : view_coverage : rejected : {rejected}")
            if rejected:
                continue
            else:
                candidate_cams.append(v_cam)
                valid_names.append(name)

        if len(candidate_cams) == 0:
            # print("[DEBUG] : view_coverage : reject : no valid candidate cameras")
            self.debug_invalid_cam(cam)
            return []
        
        sorted_idx = self.policy_view_coverage(params, candidate_cams, grid_view_dirs, view_dirs, grid_gs)
        # print(f"[DEBUG] : view_coverage : sorted_idx : {sorted_idx}")
        
        for i, idx in enumerate(sorted_idx[:self.topk]):
            selected_cam = candidate_cams[idx]
            
            with torch.no_grad():
                render_pkg = self.render_func(selected_cam, self.gaussians, self.pipe, self.bg_color)
            pred_img = render_pkg["render"]
            if "4o" in self.anchor_set: 
                selected_cam = self.get_lookat_depth(selected_cam, render_pkg)
            # Uncomment If you want to track rejected viewpoints
            # torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{selected_cam.image_name}_viewc_top{i}.png")
            
            new_traj = copy.deepcopy(traj)
            selected_cam.traj_id = new_traj._traj_id
            selected_cam.traj_order = len(new_traj)
            new_traj.add_cam(selected_cam)
            expansions.append(new_traj)
        
        return expansions[::-1] # best view comes last -> pop first !
    
    def expand_trajectory_view_coverage_refine_dfs(self, traj, _, params, grid_view_dirs, view_dirs, grid_gs=None, use_grid=False):
        """
        """
        self.clean_up()
        
        expansions = []
        
        cam = traj.last
        c2w = cam.c2w.float()
        xyz = params[0]
        
        grid_view_dirs = traj.grid_view_dirs if traj.grid_view_dirs is not None else grid_view_dirs

        if self.travel_stop_by_path(traj):
            print("[DEBUG] : dfs terminate : ", len(traj))
            traj.terminate = True
            return [traj]
        
        # if traj.check_stuck():
        #     print("[DEBUG] : dfs terminate : stuck")
        #     traj.terminate = True
        #     return [traj]
        
        c2w_np = c2w.numpy()
        candidate_c2ws, names = self.action_anchor_set(c2w_np,
                                                       depth=cam.lookat_depth if "4o" in self.anchor_set else None)
        
        candidate_cams = []
        valid_names = []
        
        for idx, (candidate_c2w, name) in enumerate(zip(candidate_c2ws, names)):
            
            name_rejected = False
            if len(traj.directions) > 0:
                name_rejected = self.camera_name_rejection(name, traj.directions[-1])
            
            if name_rejected:
                continue
            
            candidate_w2c = np.linalg.inv(candidate_c2w)    
            v_cam = self.create_vcam(cam, R=candidate_w2c[:3, :3].transpose(), T=candidate_w2c[:3, 3])
            v_cam.image_name = f"virtual_traj_{traj._traj_id}_{len(traj)}_{name}"
            rejected = self.camera_rejection(v_cam)
            # print(f"[DEBUG] : view_coverage : rejected : {rejected}")
            if rejected:
                continue
            else:
                candidate_cams.append(v_cam)
                valid_names.append(name)

        if len(candidate_cams) == 0:
            print("[DEBUG] : view_coverage : reject : no valid candidate cameras")
            self.debug_invalid_cam(cam)
            return []
        
        sorted_idx, updated_grid_view_dirs_list = self.policy_view_coverage_refine(params, candidate_cams, grid_view_dirs, view_dirs, grid_gs, use_grid)
        print(f"[DEBUG] : view_coverage_refine : sorted_idx : {sorted_idx}")
        
        if self.reverse_top:
            sorted_idx = sorted_idx[::-1]
            updated_grid_view_dirs_list = updated_grid_view_dirs_list[::-1]
        
        for i, idx in enumerate(sorted_idx[:self.topk]):
            selected_cam = candidate_cams[idx]
            updated_grid_view_dirs = updated_grid_view_dirs_list[idx]
            
            with torch.no_grad():
                render_pkg = self.render_func(selected_cam, self.gaussians, self.pipe, self.bg_color)
            pred_img = render_pkg["render"]
            if "4o" in self.anchor_set:
                selected_cam = self.get_lookat_depth(selected_cam, render_pkg)
            # Uncomment If you want to track rejected viewpoints
            # torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{selected_cam.image_name}_viewc_top{i}.png")
            
            new_traj = copy.deepcopy(traj)
            selected_cam.traj_id = new_traj._traj_id
            selected_cam.traj_order = len(new_traj)
            new_traj.grid_view_dirs = updated_grid_view_dirs
            new_traj.add_cam(selected_cam)
            expansions.append(new_traj)
        
        return expansions[::-1] # best view comes last -> pop first !
        
    
    def action_anchor_set(self, c2w_np, anchor_set=None, step=None, depth=None):
        """
        input : cam2world matrix ; numpy
        output : 
            anchor set 1 : 6 translations + 4 rotations
            anchor set 2 : 2 translations + 4 rotations
        """
        step = self.step if step is None else step * self.step
        
        anchor_set = self.anchor_set if anchor_set is None else anchor_set
        candidates_c2ws = []
        
        position = c2w_np[:3, 3]
        forward_vec = c2w_np[:3, 2]
        right_vec = c2w_np[:3, 0]
        up_vec = c2w_np[:3, 1]
        forward_vec = forward_vec / np.linalg.norm(forward_vec + 1e-15)
        right_vec = right_vec / np.linalg.norm(right_vec + 1e-15)
        up_vec = up_vec / np.linalg.norm(up_vec + 1e-15)
        angle = self.anchor_set_angle # pure camera rotation
        angle2 = self.anchor_set_angle_orbit # for orbit
        transl_names_list = ["forward", "right", "up", "backward", "left", "down"]
        rot_names_list = [f"left_{angle}", f"up_{angle}", f"right_{angle}", f"down_{angle}"]
        
        transl_names = []
        transl_dir_vecs = []
        rot_mats = []
        rot_names = []
        orbit_mats = []
        orbit_names = []
        
        if "6t" in anchor_set:
            transl_names.extend(transl_names_list)
            transl_dir_vecs.extend([forward_vec, right_vec, up_vec, -forward_vec, -right_vec, -up_vec])
        if "5t" in anchor_set:
            transl_names.extend(["forward", "right", "up", "left", "down"])
            transl_dir_vecs.extend([forward_vec, right_vec, up_vec, -right_vec, -up_vec])
        if "2t" in anchor_set:
            transl_names.extend([transl_names_list[0], transl_names_list[3]])
            transl_dir_vecs.extend([forward_vec, -forward_vec])
        if "4r" in anchor_set:
            rot_names.extend(rot_names_list)
            for angle_ in [-angle, angle]:
                right_rot = rot_c2w_right(c2w_np, -angle_)
                up_rot = rot_c2w_up(c2w_np, angle_)
                
                if self.fix_right_vec:
                    right_rot = modify_pose_rightvec(right_rot, right_vec)
                if self.fix_up_vec:
                    right_rot = modify_pose_upvec_sign(right_rot, up_vec)
                
                rot_mats.extend([right_rot, up_rot])
        if "4o" in anchor_set: # orbit style rotation
            
            if depth is None: # mean depth
                raise ValueError("depth is required for orbit style rotation")

            # FIXME:
            if self.orbit_threshold:
                # if depth > self.cam_radius: # too loose ?
                #     depth = self.cam_radius
                orbit_threshold = min(self.depth_threshold * 4, self.cam_radius)
                if depth > orbit_threshold:
                    depth = orbit_threshold
            
            position = c2w_np[:3, 3]
            lookat_point = position + forward_vec * depth
            radius = np.linalg.norm(position - lookat_point)
            sign = np.sign(forward_vec[-1])
            
            for y_angle, x_angle in zip([angle2, -angle2, 0, 0], [0, 0, angle2, -angle2]):
                x_angle_rad = np.radians(x_angle)
                y_angle_rad = np.radians(y_angle)
                
                R_azimuth = rotation_matrix_y(y_angle_rad)
                R_elevation = rotation_matrix_x(x_angle_rad)
                rotation_matrix = R_elevation @ R_azimuth
                cam_pos_init = np.array([0, 0, -radius * sign])
                cam_pos = rotation_matrix @ cam_pos_init
                
                local_c2w = create_cam2world_matrix((-position + lookat_point)[None], lookat_point[None], world_up=-up_vec)[0]
                local_pos = np.dot(local_c2w[:3, :3], cam_pos.T).T + local_c2w[:3, 3]
                cam2world = create_cam2world_keep_origin((lookat_point - local_pos)[None], local_pos[None], world_up=-up_vec)[0]
                
                # FIXME: up vector sign
                if self.fix_right_vec:
                    cam2world = modify_pose_rightvec(cam2world, right_vec)
                if self.fix_up_vec:
                    cam2world = modify_pose_upvec_sign(cam2world, up_vec)
                
                orbit_mats.append(cam2world)
                orbit_names.append(f"orbit_{y_angle}_{x_angle}")
                
        for name, dir_vec in zip(transl_names, transl_dir_vecs):
            new_position = position + step * dir_vec
            new_c2w = c2w_np.copy()
            new_c2w[:3, 3] = new_position
            candidates_c2ws.append(new_c2w)
            
        for name, rot_mat in zip(rot_names, rot_mats):
            candidates_c2ws.append(rot_mat)
            # new_c2w = c2w_np.copy()
            # new_c2w[:3, :3] = rot_mat
            # candidates_c2ws.append(new_c2w)
        
        for name, orbit_mat in zip(orbit_names, orbit_mats):
            candidates_c2ws.append(orbit_mat)   
        
        candidate_names = transl_names + rot_names + orbit_names
        return candidates_c2ws, candidate_names

    def policy_nbvs(self, params, candidate_cams, I_train):
        
        H_candidates = []
        # renders = []
        
        for cam in candidate_cams:
            render_fisher_pkg = self.render_fisher(cam, self.gaussians, self.pipe, self.bg_color)
            pred_img = render_fisher_pkg["render"]
            # renders.append(pred_img)
            pred_img.backward(gradient=torch.ones_like(pred_img))
            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
            H_candidates.append(cur_H.cpu())
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        sorted_idx = self.next_best_view_selection(H_candidates, I_train)
        return sorted_idx

    def policy_view_coverage(self, params, candidate_cams, grid_view_dirs, view_dirs, grid_gs=None):
        """
        grid_view_dirs : (N_grid, N_view)
        view_dirs : (N_view, 3)
        (1) compute angle similarity between view_dirs and cam_forward
        (2) select the view with the highest angle similarity
        (3) update grid_view_dirs by setting the selected view to 1
        (4) compute the gain of the selected view : next_grid_view_dirs.sum() - grid_view_dirs.sum()
            # trick : can be replaced by counting new 1s.
        (5) return the sorted index of the selected view
        """
        gains = []
        
        for cam in candidate_cams:
            render_pkg = self.render_func(cam, self.gaussians if grid_gs is None else grid_gs, self.pipe, self.bg_color)
            
            mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy()
            
            cam_forward = cam.c2w[:3, 2].numpy()
            angle_sim = view_dirs @ cam_forward
            
            idx = np.argmax(angle_sim)
            gain = (grid_view_dirs[mask_visible, idx] == 0).sum()
            gains.append(gain)
        
        print(f"[DEBUG] : view_coverage : gains : {gains}")
        
        sorted_idx = np.argsort(gains)[::-1]
        return sorted_idx

    # TODO: add visiblity-based weight
    @torch.no_grad()
    def policy_view_coverage_refine(self, params, candidate_cams, grid_view_dirs, view_dirs, grid_gs=None, use_grid=False):
        """
        grid_view_dirs : (N_grid, N_view)
        view_dirs : (N_view, 3)
        if use_grid:
            grid_gs = track_gs
        else:
            grid_gs = self.gaussians to get gaussians positions
        """
        gains = []
        new_grid_view_dirs_list = []
        render_gs = grid_gs if use_grid else self.gaussians
        grid_gs = grid_gs if use_grid else self.gaussians
        
        for cam in candidate_cams:
            render_pkg = self.render_func(cam, render_gs, self.pipe, self.bg_color)
            
            mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy()
            if self.gs_proj:
                to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
                xyz = grid_gs.get_xyz[mask_visible].detach()
                pts3d_homo = to_homo(xyz)
                pts3d_cam = pts3d_homo @ cam.world_view_transform[:, :3] # (N_mask, 3)
                cam_view_dirs = pts3d_cam - cam.c2w[:3, 3].to(pts3d_cam.device) # (N_mask, 3)
                cam_view_dirs = cam_view_dirs.cpu().numpy()
                view_dirs = view_dirs @ cam.world_view_transform[:3, :3].cpu().numpy().transpose() # (M, 3) # ???
            else:
                cam_view_dirs = grid_gs.get_xyz[mask_visible].detach().cpu().numpy() - cam.c2w[:3, 3].numpy() # (N_gs, 3)
            cam_view_dirs = cam_view_dirs / np.linalg.norm(cam_view_dirs, axis=1)[:, np.newaxis] # (N_gs, 3)
            # get angle between cam_view_dirs and view_dirs
            angle_sim = cam_view_dirs @ view_dirs.T # (N_gs, M)
            # get index of view_dirs
            idx = np.argmax(angle_sim, axis=1)
            new_grid_view_dirs = grid_view_dirs.copy()
            new_grid_view_dirs[mask_visible, idx] = 1
            new_grid_view_dirs_list.append(new_grid_view_dirs)
            
            if self.gain_weight != "":
                if self.gain_weight == "visibility":
                    transmittance = render_pkg["T"][mask_visible] # weight
                    gs_counter = render_pkg["gaussian_pixel_counter"][mask_visible]
                    transmittance = torch.nan_to_num(transmittance / gs_counter, nan=0.0)
                    visibility = self.t_cache[mask_visible]
                    # FIXME: wrong formula
                    weight = transmittance * (1 - visibility) # to look at unseen regions
                    gain = (new_grid_view_dirs.sum() - grid_view_dirs.sum()) * weight.mean().item()
                else:
                    gain = (new_grid_view_dirs.sum() - grid_view_dirs.sum())
            else:
                gain = (new_grid_view_dirs.sum() - grid_view_dirs.sum())
            gains.append(gain)
        
        print(f"[DEBUG] : view_coverage : gains : {gains}")
        
        sorted_idx = np.argsort(gains)[::-1]
        updated_grid_view_dirs_list = []
        for idx in sorted_idx:
            updated_grid_view_dirs = new_grid_view_dirs_list[idx]
            updated_grid_view_dirs_list.append(updated_grid_view_dirs)
        return sorted_idx, updated_grid_view_dirs_list
    
    def camera_rejection(self, cam):
        
        if self.skip_rejection:
            return False
        
        def save_rejected_cam(cam, suffix=""):
            with torch.no_grad():
                render_pkg = self.render_func(cam, self.gaussians, self.pipe, self.bg_color)
                pred_img = render_pkg["render"]
                # Uncomment If you want to track rejected viewpoints
                # torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{cam.image_name}_reject_{suffix}.png")
        
        if self.skip_freespace_rejection:
            is_free = True
        else:
            is_free = self.voxel_check(cam.c2w[:3, 3].numpy(), self.search_space, self.labels, radius=self.depth_threshold)
            if not is_free:
                save_rejected_cam(cam, suffix="free")
                return True
            
        render_reject_pkg = self.render_func(cam, self.gaussians, self.pipe, self.bg_filter, skip_clamp=True)
        reject_bg, bg_mask = self.reject_cam_by_bg(render_reject_pkg)
        if reject_bg:
            save_rejected_cam(cam, suffix="bg")
            return True
        else:
            reject_depth, depth_mask = self.reject_cam_by_depth(render_reject_pkg, bg_mask)
            if reject_depth:
                save_rejected_cam(cam, suffix="depth")
                return True
            else:
                return False
        
    def occupancy_track(self, voxel_grid, scene_mesh, dist_threshold=None, distance=False, use_frontier=False):
        # 0. settings
        track_gs = copy.deepcopy(self.gaussians)
        if dist_threshold is None:
            dist_threshold = self.depth_threshold
        
        # examine gaussians close to cameras
        xyz = track_gs.get_xyz
        gs_threshold = dist_threshold * 0.1

        # del_idx = torch.zeros(len(xyz), device=self.device)
        # for cam in self.train_cameras:
        #     pos = cam.c2w[:3, 3].to(xyz.device)
        #     dist = torch.norm(xyz - pos, dim=1)
        #     min_dist = dist.min()
        #     print("[DEBUG] : min_dist = ", min_dist)
        # del_idx = del_idx > 0
        
        device = self.device
        # original
        # track_gs._opacity = track_gs._opacity * 1e6 # force to be near 1 ~= solid mesh
        # modified
        track_gs._opacity = torch.where(
            track_gs._opacity > 0.5,
            track_gs._opacity * 1e6,
            track_gs._opacity * 1e-6
        )
        # [INFO] : free : 97698, occupied : 164446, unexplored : 110550
        
        # 1. voxel grid cell -> gaussian points ; random color but zero opacity (use inverse sigmoid for initialization)
        track_pts = torch.from_numpy(voxel_grid).float().to(device)
        idx_start = len(self.gaussians.get_xyz)
        track_features_dc = torch.randn((len(track_pts), 1, 3), device=device)
        track_features_rest = torch.zeros((len(track_pts), 15, 3), device=device)
        # track_opacity = inverse_sigmoid(torch.zeros_like(track_pts[:, 0]))[..., None] # too small alpha -> filterd ?
        track_opacity = inverse_sigmoid(0.005 * torch.ones_like(track_pts[:, 0]))[..., None]
        track_rotation = torch.zeros((len(track_pts), 4), device=device)
        tmpxyz = torch.cat((track_pts, self.gaussians.get_xyz), dim=0)
        dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
        dist2 = dist2[:track_pts.shape[0]]
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        track_scales = torch.clamp(scales, -10, 1.0)
        
        # 2. result of 1 -> add_gaussians
        track_gs.densification_postfix(track_pts, track_features_dc, track_features_rest, track_opacity, track_scales, track_rotation)
        
        # 3. render & track T of new gaussians
        T_cache = torch.zeros(len(self.train_cameras), (len(track_pts)), device=device)
        T_count = torch.zeros((len(self.train_cameras), len(track_pts)), device=device)
        
        for idx, cam in enumerate(self.train_cameras):
            with torch.no_grad():
                render_pkg = self.render_func(cam, track_gs, self.pipe, self.bg_color)
            # rgb = render_pkg["render"]
            # torchvision.utils.save_image(rgb, f"debugging/rgb_{idx}.png") # working well ...
            out_T = render_pkg["T"]
            gs_counter = render_pkg["gaussian_pixel_counter"]
            visibility = render_pkg["visibility_filter"]
            
            T_cache[idx] = torch.nan_to_num(out_T[idx_start:] / gs_counter[idx_start:], nan=0.0)
            T_count[idx] = visibility[idx_start:]
            # print(T_cache[idx], T_count[idx])
            
        # avg_T = avg_T.sum(dim=0)
        # avg_T_count = avg_T_count.sum(dim=0)
        # # import pdb; pdb.set_trace()
        # # 4. Given threshold, label voxel grid cell as occupied or free space
        # track_T = avg_T / avg_T_count
        # print(avg_track_T)
        # print("[DEBUG] : T_cache = ", T_cache)
        # print("[DEBUG] : T_count = ", T_count)
        selected_T, _ = torch.topk(T_cache, k=min(5, len(T_cache)), dim=0)
        # import pdb; pdb.set_trace()
        selected_T = selected_T.mean(dim=0)
        track_T = selected_T
        # torch inf to -1e6
        track_T = torch.where(track_T == float('-inf'), -1e6, track_T)
        
        # print("[DEBUG] : track_T = ", track_T)
        
        # track_T = repeat(selected_T, "n -> n c", c=3)
        # print("[DEBUG] : track_T.max(dim=0) : ", track_T.max(dim=0))
        
        # FIXME: transmittance 가 0 인데 occupied 로 판단되는 경우가 있음... maybe floater ?
        # high transmittance -> free, low transmittance -> occupied
        unexplored = (T_count.sum(dim=0) == 0 + (track_T < 0)) > 0 # never visited by any training camera
        # free = (track_T >= threshold) * ~unexplored # at least one camera has transmittance >= threshold
        occupied = (0 <= track_T) * (track_T < self.occ_threshold) * ~unexplored # at least one camera has transmittance < threshold
        
        # compute distance between mesh and voxel grid
        mesh_pts = torch.from_numpy(np.asarray(scene_mesh.vertices)).float().to(self.device)
        mesh_pts = mesh_pts.unsqueeze(0) # (1, M, 3)
        
        voxel_grid = torch.from_numpy(voxel_grid).float().to(self.device)[None] # (1, N, 3)
        
        # chunk = 16384
        chunk = 4096
        dist = []
        for i in range(0, voxel_grid.shape[1], chunk):
            cur_voxel_grid = voxel_grid[:, i:i+chunk, :]
            cur_dist = torch.cdist(cur_voxel_grid, mesh_pts, compute_mode="donot_use_mm_for_euclid_dist").squeeze().min(dim=1).values
            dist.append(cur_dist)
        dist = torch.cat(dist, dim=0)
        # print("[DEBUG] : dist = ", dist)
        # FIXME: main bottleneck!!
        if self.mesh_dist_threshold != "":
            if type(self.mesh_dist_threshold) == float: #1.0
                dist_threshold = self.depth_threshold * self.mesh_dist_threshold
            elif "voxel_size" in self.mesh_dist_threshold:
                multiplier = float(self.mesh_dist_threshold.split("_")[-1])
                dist_threshold = self.voxel_size * multiplier
            elif "mix" in self.mesh_dist_threshold:
                # for outdoor / indoor
                depth_val = float(self.mesh_dist_threshold.split("_")[1])
                voxel_val = float(self.mesh_dist_threshold.split("_")[2])
                dist_threshold = min(depth_val * self.depth_threshold, voxel_val * self.voxel_size)
                print(f"[INFO] : {depth_val} {voxel_val} {self.depth_threshold} {self.voxel_size}")
                print(f"[INFO] : {depth_val * self.depth_threshold} vs {voxel_val * self.voxel_size} >> dist_threshold = {dist_threshold}")
            occupied_dist = dist < dist_threshold
            
            if distance:
                occupied = (occupied + occupied_dist) > 0 # OR
            
        # dist_gs = torch.cdist(voxel_grid.cpu(), self.gaussians.get_xyz.cpu()).squeeze().min(dim=1).values
        # Instead, use chunk
        # dist_gs = []
        # chunk = 1000
        # gs_chunk = 1000
        # for i in range(0, len(voxel_grid), chunk):
        #     cur_voxel_grid = voxel_grid[:, i:i+chunk, :]
        #     cur_dist_gs = []
        #     for j in range(0, len(self.gaussians.get_xyz), gs_chunk):
        #         cur_gs = self.gaussians.get_xyz[j:j+gs_chunk]
        #         cur_dist = torch.cdist(cur_voxel_grid, cur_gs).squeeze().min(dim=1).values.cpu()
        #         cur_dist_gs.append(cur_dist)
        #     dist_gs.append(torch.cat(cur_dist_gs, dim=0))
        # dist_gs = torch.cat(dist_gs, dim=0)
        # occupied_gs_dist = dist_gs.cuda() < gs_threshold
        # occupied = (occupied + occupied_dist + occupied_gs_dist) > 0 # OR
        
        # original (try 3 (~= 2-2))
        free = (occupied + unexplored) == 0 # NOT occupied or unexplored
        
        print(f"[INFO] : free : {free.sum()}, occupied : {occupied.sum()}, unexplored : {unexplored.sum()}")
        # print(f"[DEBUG] : free = {free}")
        # print(f"[DEBUG] : occupied = {occupied}")
        # print(f"[DEBUG] : unexplored = {unexplored}")
        
        free_voxel_indices = free.nonzero().squeeze().cpu().numpy()
        occupied_voxel_indices = occupied.nonzero().squeeze().cpu().numpy()
        unexplored_voxel_indices = unexplored.nonzero().squeeze().cpu().numpy()
        
        if use_frontier > 0:
            free_voxel_grids = voxel_grid[0][free] # (N_free, 3)
            occupied_voxel_grids = voxel_grid[0][occupied] # (N_occupied, 3)
            chunk = 1024
            mask = []
            for i in range(0, len(free_voxel_grids), chunk):
                cur_free_voxel_grids = free_voxel_grids[i:i+chunk]
                cur_mask = torch.cdist(cur_free_voxel_grids, occupied_voxel_grids).min(dim=1).values < self.voxel_size * use_frontier
                mask.append(cur_mask)
            mask = torch.cat(mask, dim=0)
            frontier_sub_inds = mask.nonzero().squeeze(-1)
            # mask = torch.cdist(free_voxel_grids, occupied_voxel_grids).min(dim=1).values < self.voxel_size * use_frontier
            # frontier_sub_inds = mask.nonzero().squeeze(-1)
            frontier_voxel_indices = free_voxel_indices[frontier_sub_inds.cpu().numpy()]
            # TODO: get frontier voxel indices
            print(f"[INFO] : frontier : {frontier_voxel_indices.shape}")    
        else:
            frontier_voxel_indices = None
        
        # print(f"[DEBUG] : free_voxel_indices : {free_voxel_indices}, occupied_voxel_indices : {occupied_voxel_indices}, unexplored_voxel_indices : {unexplored_voxel_indices}")
        del track_gs
        del T_cache
        del T_count
        
        return free_voxel_indices, occupied_voxel_indices, unexplored_voxel_indices, frontier_voxel_indices
        
    def create_vcam(self, camera0, 
                    R=None, T=None, 
                    image_name="virtual", 
                    cost=0, 
                    width=None, height=None, 
                    image=None):
        
        cp = self.cam_params
        mp = self.mask_params
        
        R = camera0.R if R is None else R
        T = camera0.T if T is None else T
        
        ori_h, ori_w = camera0.image_height, camera0.image_width
        # self.v_resolution = target resolution
        # camera0.FoVx , camera0.FoVy
        
        adjusted_fovx = width / ori_w * camera0.FoVx if width is not None else camera0.FoVx
        adjusted_fovy = height / ori_h * camera0.FoVy if height is not None else camera0.FoVy
        
        v_cam = Camera(colmap_id=camera0.colmap_id, R=R, T=T,
                        FoVx=adjusted_fovx, FoVy=adjusted_fovy, 
                        image=camera0.original_image.cpu() if camera0.original_image is not None else None, 
                        image_name=image_name,
                        gt_alpha_mask=None,
                        uid=camera0.uid, trans=camera0.trans, # not used
                        scale=camera0.scale, data_device=camera0.data_device, # not used
                        override_height=self.image_height if self.image_height > 0 else height,
                        override_width=self.image_width if self.image_width > 0 else width)
        # if isinstance(image, bool):
        #     v_cam.original_image = v_cam.original_image.cpu() if image else None
        # else:
        #     v_cam.original_image = image   # register new
        v_cam.original_image = None
        v_cam.virtual = True
        
        if mp.replace_gt: # used with aug_mask
            v_cam.replace_img = camera0.original_image
        return v_cam

    def get_lookat_depth(self, cam, render_pkg=None):
        if render_pkg is None:
            with torch.no_grad():
                render_pkg = self.render_func(cam, self.gaussians, self.pipe, self.bg_color)        
        pred_depth = render_pkg["depth"]
        h, w = pred_depth.shape[-2:]
        # FIXME: instead mean , use median ?!
        lookat_depth = pred_depth[0, int(h//4):int(h*3//4), int(w//4):int(w*3//4)].mean().item()
        cam.lookat_depth = lookat_depth
        return cam

    def travel_stop_by_path(self, traj):
        
        for stop in self.travel_stop:
            if "length" in stop:
                length = int(stop.split("-")[1])
                if len(traj) >= length:
                    return True
            elif "path" in stop:
                path = int(stop.split("-")[1])
                threshold = self.voxel_size * self.voxel_resolution * path
                if traj.get_path_length() >= threshold:
                    return True

        return False
        
    def reject_cam_by_depth(self, render_pkg, bg_mask, threshold=0.5):
        # threshold_depth = self.space_unit * self.depth_threshold
        
        depth = render_pkg["depth"] # ? : how to deal empty space depth
        depth[bg_mask[None]] = self.space_unit * self.voxel_resolution * 2 # arbitrary large value
        # depth_mask = depth < threshold_depth # near-zero depth -> reject
        # depth_mask = depth < self.depth_threshold
        
        if "voxel_size" in self.reject_depth:
            depth_mask = depth < self.voxel_size
        elif "mix" in self.reject_depth:
            depth_val = self.reject_depth.split("_")[1]
            voxel_val = self.reject_depth.split("_")[2]
            depth_threshold = min(float(depth_val) * self.depth_threshold, float(voxel_val) * self.voxel_size)
            depth_mask = depth < depth_threshold
        else:
            depth_mask = depth < self.depth_threshold
        
        valid = depth_mask.sum() / depth_mask.numel()
        # print("[DEBUG] : reject_cam_by_depth : ", valid, depth_mask.sum(), depth_mask.numel(), depth.mean(), depth.max(), depth.min())
        return valid > threshold, depth
    
    def reject_cam_by_bg(self, render_pkg, threshold=None):
        if threshold is None:
            threshold = self.bg_reject_ratio
        
        pred_img = render_pkg["render"]
        bg_mask = pred_img.mean(dim=0) > 9.0 # (10 - 0.1) # fixed value here ?!
        valid = bg_mask.sum() / bg_mask.numel()
        
        return valid > threshold, bg_mask
    
    def get_H_train(self, params):
        
        H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)
        for cam in self.train_cameras:
            render_pkg = self.render_fisher(cam, self.gaussians, self.pipe, self.bg_color)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))
            
            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
            H_train += cur_H
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            
        return H_train

    def next_best_view_selection(self, candidates, I_train):
        scores = np.array([torch.sum(cur_H * I_train).item() for cur_H in candidates])
        # selected_idx = np.argsort(scores)[-n_views:].tolist()  # Get indices of top n_views scores
        # return selected_idx, [scores[i] for i in selected_idx]
        return np.argsort(scores)[::-1]

    def clean_up(self):
        torch.cuda.empty_cache() # essential ; due to recursive manner
        gc.collect()

    def debug_invalid_cam(self, cam):
        with torch.no_grad():
            render_pkg = self.render_func(cam, self.gaussians, self.pipe, self.bg_color)
        pred_img = render_pkg["render"]
        torchvision.utils.save_image(pred_img, f"{self.dbg_dir}/{cam.image_name}_reject_occupied.png")
    
    def camera_name_rejection(self, name, last_direction):
        if last_direction == "up":
            if "down" == name:
                return True
        elif last_direction == "down":
            if "up" == name:
                return True
        elif last_direction == "left":
            if "right" == name:
                return True
        elif last_direction == "right":
            if "left" == name:
                return True
        elif last_direction == "forward":
            if "backward" == name:
                return True
        elif last_direction == "backward":
            if "forward" == name:
                return True
        elif last_direction == "up_15":
            if "down_15" == name:
                return True
        elif last_direction == "down_15":
            if "up_15" == name:
                return True
        elif last_direction == "left_15":
            if "right_15" == name:
                return True
        elif last_direction == "right_15":
            if "left_15" == name:
                return True
        return False
    
    # FIX : radius --> voxel size
    def in_free_space(self, point, free_voxels, labels, radius=None):
        """
        Check if 'point' is inside a voxel labeled 'free'.
        e.g. do a nearest neighbor search in free_voxels.
        """
        if radius is None:
            radius = self.voxel_size
        dist = np.linalg.norm(free_voxels - point, axis=1)
        nearest_idx = np.argmin(dist)
        min_dist = dist[nearest_idx]
        if min_dist < radius and labels[nearest_idx] == 'free':
            return True
        return False

class Trajectory:
    _traj_id = 0
    
    def __init__(self, init_cam=None, root=""):
        self.path = []
        self.directions = []
        self.terminate = False
        Trajectory._traj_id += 1
        self._traj_id = Trajectory._traj_id
        self.grid_view_dirs = None
        
        if init_cam is not None:
            self.path.append(init_cam)
            
        if root != "":
            self.root = root
        else:
            if len(self.path) > 0:
                self.root = self.path[0].image_name
            else:
                self.root = ""
    
    def add_cam(self, cam):
        cam.ref_img_name = self.root
        self.path.append(cam)
        for name in ["left_15", "up_15", "right_15", "down_15", "forward", "right", "up", "backward", "left", "down", ]:
            if name in cam.image_name:
                self.directions.append(name)
                break

    def get_path(self): # return camera positions
        return [cam.c2w[:3, :3] for cam in self.path]
    
    def get_path_length(self): # sum of line semgents
        path_length = 0
        for i in range(len(self.path) - 1):
            pos1 = self.path[i].c2w[:3, 3]
            pos2 = self.path[i+1].c2w[:3, 3]
            path_length += np.linalg.norm(pos1 - pos2)
        return path_length
    
    def __len__(self):
        return len(self.path)
    
    @property
    def last(self):
        return self.path[-1]
    
    def debug(self, length):
        # copy init cam to self.path. This is only for debugging.
        for i in range(length):
            self.path.append(self.path[0])