import math
import numpy as np
import torch
from typing import List
from scene.cameras import Camera
from utils.camera_utils import camera_to_JSON
from utils.graphics_utils import fov2focal, focal2fov
from utils.general_utils import render_based_on_mask_type
import torchvision
import os
from tqdm import tqdm
from collections import defaultdict
import json


_EPS = np.finfo(float).eps * 4.0

"""
All operations are computed at c2w
"""
class CameraGenerator:
    def __init__(self, 
                 scene, 
                 cam_params, 
                 diff_params,
                 mask_params, 
                 renderer, 
                 gaussians, 
                 pipe,
                 opt,
                 bg,
                 args):
        """
        scene : class Scene
        data : ModelParams
        """
        self.cam_params = cam_params
        self.diff_params = diff_params
        self.mask_params = mask_params
        
        self.renderer = renderer
        self.gaussians = gaussians
        self.pipe = pipe
        self.opt = opt
        self.bg = bg

        self.train_cameras = scene.getTrainCameras() # dict
        
        self.image_width = self.train_cameras[0].image_width
        self.image_height = self.train_cameras[0].image_height
        
        self.test_cameras = scene.getTestCameras() # dict
        
        print(f"[INFO] : CameraGenerator : {len(self.train_cameras)} cameras (GT)")
        print(f"[INFO] : CameraGenerator : {len(self.test_cameras)} cameras (Test)")
        self.camera_extent = scene.cameras_extent
        self.sampling_strategy = cam_params.sampling_strategy
        
        self.v_resolution = diff_params.input_resolution

        # preprocess
        self.train_cameras = self.rearrange_cams(self.train_cameras)
        self.test_cameras = self.rearrange_cams(self.test_cameras)
        self.train_cameras_dict = {cam.image_name: cam for cam in self.train_cameras}
        self.test_cameras_dict = {cam.image_name: cam for cam in self.test_cameras}
        
        self.train_cameras_all_trans = self.all_cam_trans(self.train_cameras)

        self.scene_name = args.source_path.split('/')[-1]
        self.save_path = scene.model_path + "/vcam_traj_images"
        os.makedirs(self.save_path, exist_ok=True)
        print("[DEBUG] : ", self.scene_name)
        self.skip_inspect = False
        
        self.video_seqs = {}
        
        self.cam_radius = None

    def rearrange_cams(self, cameras:List):
        # image name order
        # TODO
        return sorted(cameras, key=lambda cam: cam.image_name)

    def all_cam_trans(self, cameras):
        all_trans = np.zeros((len(cameras), 3))
        for idx, cam in enumerate(cameras):
            c2w = cam.c2w.numpy()
            all_trans[idx] = c2w[:3, 3]
        return all_trans

    def prepare_diffusion_cameras(self, cameras, n_updates:int, **kwargs):
        if self.cam_params.reference_strategy == "video-sequence": # for 3DGS-Enhancer
            if self.video_seqs is {}:
                raise ValueError("[ERROR] : video_seqs is empty")
            
            batch = n_updates // 16
            if n_updates % 16 != 0:
                print(f"[WARNING] : n_updates % 16 != 0 ; set batch = {batch} n_updates = {n_updates}")
            
            # filtering
            last_iter_set = sorted(self.video_seqs.items(), key=lambda seq: seq[1].last_iter)
            candidates = last_iter_set[:batch * 2]
            candidates = {key: seq for key, seq in candidates}
            
            cost_set = sorted(candidates.items(), key=lambda seq: seq[1].cost)
            candidates = {key: seq for key, seq in cost_set[:batch]}
            
            cond_frame_idx_set = []
            # inject train camera
            print("[DEBUG] : len(train_cameras) = ", len(self.train_cameras))
            
            for key, vid_seq in candidates.items():
                # train_cam_idx = vid_seq.train_cam_idx
                closest_idx = vid_seq.closest_idx
                # print("[DEBUG] : key = ", key, "train_cam_idx = ", train_cam_idx, "closest_idx = ", closest_idx, "camera_set = ", len(vid_seq.camera_set))
                # vid_seq.camera_set[closest_idx] = self.train_cameras[train_cam_idx]
                vid_seq.camera_set[closest_idx] = self.train_cameras_dict[vid_seq.ref_img_name]
                vid_seq.camera_idx_set.pop(closest_idx)
            return self.wrap_video_frames(candidates, **kwargs)

        else:
            raise NotImplementedError(f"[ERROR] : virtual_cam_utils.py - CameraGenerator - prepare_diffusion_cameras")

    def prepare_video_frames(self, dataset, search=False):
        # dataset = v_cam_set
        # 1. collect frames 
        seq_set = {}
        seq_idx_set = {}
        
        print("[DEBUG] : prepare_video_frames : sampling_strategy = ", self.cam_params.sampling_strategy)
        
        for idx, cam in enumerate(dataset):
            key = cam.traj_id

            if key in seq_set:
                seq_set[key].append(cam)
                seq_idx_set[key].append(idx)
            else:
                seq_set[key] = [cam]
                seq_idx_set[key] = [idx]

        print("[DEBUG] : seq_set : ", [len(seq_set[key]) for key in seq_set.keys()])
        print("[DEBUG] : seq_set ref_img_name : ", [seq_set[key][0].ref_img_name for key in seq_set.keys()])
        
        # 2. sort frames
        for key in seq_set.keys():
            pairs = list(zip(seq_set[key], seq_idx_set[key]))
            pairs = sorted(pairs, key=lambda p: p[0].traj_order)
            seq_set[key], seq_idx_set[key] = zip(*pairs)
            
            seq_set[key] = list(seq_set[key])
            seq_idx_set[key] = list(seq_idx_set[key])

        # 3. wrap VideoSequence
        seq_id = 0
        for key in seq_set.keys():
            seq = seq_set[key]
            seq_idx = seq_idx_set[key]
            
            vid_len = 16
            if len(seq) > vid_len:
                raise ValueError(f"[ERROR] : prepare_video_frames : sequence length {len(seq)} exceeds video length {vid_len}")
            
            self.video_seqs[key] = VideoSequence(seq, seq_idx, self.cam_params, seq_id)
            seq_id += 1

    def create_vcam(self, camera0, 
                    R=None, T=None, 
                    image_name="virtual", 
                    cost=0, 
                    width=None, height=None, 
                    image=None):
        """
        Note : R = c2w ; T = w2c
        if image is None = virtual cam / else used for split
        """
        cp = self.cam_params
        mp = self.mask_params
        
        R = camera0.R if R is None else R
        T = camera0.T if T is None else T
        
        # ori_h, ori_w = camera0.original_image.shape[-2:]
        
        # self.v_resolution = target resolution
        # camera0.FoVx , camera0.FoVy
        ori_h, ori_w = self.image_height, self.image_width
        
        adjusted_fovx = width / ori_w * camera0.FoVx if width is not None else camera0.FoVx
        adjusted_fovy = height / ori_h * camera0.FoVy if height is not None else camera0.FoVy

        v_cam = Camera(colmap_id=camera0.colmap_id, R=R, T=T,
                        FoVx=adjusted_fovx, FoVy=adjusted_fovy, 
                        image=camera0.original_image if image is None else image, 
                        image_name=image_name,
                        gt_alpha_mask=None,
                        uid=camera0.uid, trans=camera0.trans, # not used
                        scale=camera0.scale, data_device=camera0.data_device, # not used
                        override_height=self.image_height,
                        override_width=self.image_width,  
                        cost=cost
                        )
        if isinstance(image, bool):
            v_cam.original_image = v_cam.original_image if image else None
        else:
            v_cam.original_image = image    # register new
        v_cam.virtual = True if image is None else False # 

        if mp.replace_gt: # used with aug_mask
            v_cam.replace_img = camera0.original_image

        return v_cam
    
    def inspect_vcam(self, vcam, suffix=''):
        with torch.no_grad():
            render_pkg = self.renderer(vcam, self.gaussians, self.pipe, self.bg)
            rendering = render_pkg["render"]
            os.makedirs("debugging", exist_ok=True)
            torchvision.utils.save_image(rendering, os.path.join(self.save_path, f"{self.scene_name}_{vcam.image_name}{suffix}.png"))

    def wrap_video_frames(self, vid_seq_dict, **kwargs):
        # v_cam_set : list of Camera
        # cond_frame_idx_set : list of int
        # return diff_input_pkg : dict {"renders": tensor, "cond_frame_idx": list or tensor, "render_pose": tensor}
        
        renders_batch = []
        poses_batch = []
        guidance_batch = []
        names_batch = []
        idx_set_batch = []
        cond_frame_idx_set = []
        pixel_gs_counters = []
        visibility_filters = []
        avgdepths = []
        invdepths = []
        
        kwargs["pair_cameras"] = self.train_cameras
        kwargs["renderer"] = self.renderer
            
        for key, vid_seq in vid_seq_dict.items():
            poses = []
            names = []
            kwargs["query_cameras"] = vid_seq.camera_set
            with torch.no_grad():
                render_pkg = render_based_on_mask_type(**kwargs)
                imgs = render_pkg["pred_img"]
                guidances = render_pkg["uncertainty_map"]
                
                pixel_gs_counter = render_pkg["pixel_gaussian_counter"] # list
                visibility_filter = render_pkg["visibility_filter"] # list
                
                avgdepth = render_pkg["avgdepth"] # List[tensor]
                invdepth = render_pkg["invdepth"] # List[tensor]

            idx_set_batch.extend(vid_seq.camera_idx_set)
            for idx, v_cam in enumerate(vid_seq.camera_set):
                poses.append(v_cam.c2w)
                names.append(v_cam.image_name)
            imgs = torch.stack(imgs)
            cond_frame_idx_set.append([vid_seq.closest_idx])
            
            pixel_gs_counter = torch.stack(pixel_gs_counter)
            visibility_filter = torch.stack(visibility_filter)
            avgdepth = torch.stack(avgdepth).squeeze() # (T, H, W)
            invdepth = torch.stack(invdepth).squeeze() # (T, H, W)
            
            guidances = torch.stack(guidances)[:, None]
            poses = torch.stack(poses)
            
            names_batch.append(names)
            # replace closest camera with training viewpoint
            nearest_cam = vid_seq.camera_set[vid_seq.closest_idx]
            imgs[vid_seq.closest_idx] = nearest_cam.original_image
            poses[vid_seq.closest_idx] = nearest_cam.c2w
            if nearest_cam.invdepthmap is not None:
                invdepth[vid_seq.closest_idx] = nearest_cam.invdepthmap
            
            # store cpu tensors instead of GPU RAM
            renders_batch.append(imgs.cpu())
            poses_batch.append(poses.cpu())
            guidance_batch.append(guidances.cpu())
            pixel_gs_counters.append(pixel_gs_counter.cpu())
            visibility_filters.append(visibility_filter.cpu())
            avgdepths.append(avgdepth.cpu())
            invdepths.append(invdepth.cpu())

        renders_batch = torch.stack(renders_batch)
        poses_batch = torch.stack(poses_batch)
        guidance_batch = torch.stack(guidance_batch)
        avgdepths = torch.stack(avgdepths)
        invdepths = torch.stack(invdepths)

        return {"renders": renders_batch, "cond_frame_idx": cond_frame_idx_set, "render_pose": poses_batch, 
                "guidance": guidance_batch, "names": names_batch, "idx_set": idx_set_batch,
                "pixel_gs_counters": pixel_gs_counters, "visibility_filters": visibility_filters,
                "avgdepths": avgdepths, "invdepths": invdepths}

    def wrap_trajectories(self, trajectories, vid_length=16, skip_inspect=None, search=False):
        """
        wrap trajectories into dataset
        """
        dataset = []
        for traj in trajectories:
            # import pdb; pdb.set_trace()
            
            cam_set = traj.path
            print("[DEBUG] : cam_set = ", cam_set)
            
            cam_c2ws = [cam.c2w for cam in cam_set]
            cam_dist_from_first = [np.linalg.norm(cam_c2ws[i][:3, 3] - cam_c2ws[0][:3, 3]) for i in range(len(cam_c2ws))]
            cam_dist_max = max(cam_dist_from_first)
            cam_dist_min = min(cam_dist_from_first)

            # make smooth trajectory (arbitrary shape)
            # sample vid_length along the trajectory
            smoothed = []
            total_segments = len(cam_c2ws) - 1
            for i in range(vid_length):
                p = i * total_segments / (vid_length - 1)
                idx = int(np.floor(p))
                t = p - idx
                
                if idx >= total_segments:
                    idx = total_segments - 1
                    t = 1.0
                    
                c2w_new = self.slerp_c2w(cam_c2ws[idx], cam_c2ws[idx + 1], t)
                w2c_new = np.linalg.inv(c2w_new)
                if i > 0:
                    new_name = cam_set[i].image_name.replace("traj", "straj") + f"_t{p:.2f}"
                else:
                    new_name = cam_set[i].image_name + f"_t{p:.2f}"
                print(f"[DEBUG] : {i}th new_name = ", new_name)
                cam_new = self.create_vcam(cam_set[i], 
                                           R=c2w_new[:3, :3],
                                           T=w2c_new[:3, 3], 
                                           image_name=new_name)
                cam_new.traj_order = cam_set[i].traj_order
                cam_new.traj_id = cam_set[i].traj_id
                cam_new.ref_img_name = cam_set[i].ref_img_name
                if self.opt.image_confidence == "cost2":
                    cam_new.cost = (cam_dist_from_first[i] / self.cam_radius).clamp(0, 1)
                elif self.opt.image_confidence == "cost":
                    cam_new.cost = (cam_dist_from_first[i] / cam_dist_max)
                # TODO: add distance for ablation !
                smoothed.append(cam_new)
            dataset.extend(smoothed)
            
        if self.diff_params.wrap_seq:
            self.prepare_video_frames(dataset)
        
        if skip_inspect is None:
            skip_inspect = self.skip_inspect
        
        if not skip_inspect:
            for v_cam in tqdm(dataset, desc="[INFO] : inspect virtual cameras"):
                self.inspect_vcam(v_cam)
            
        return dataset
    
    def slerp_c2w(self, c2w1, c2w2, t):
        
        R1 = c2w1[:3, :3]
        R2 = c2w2[:3, :3]
        q1 = quaternion_from_matrix(R1)
        q2 = quaternion_from_matrix(R2)
        q = quaternion_slerp(q1, q2, t)
        R = quaternion_matrix(q)[:3, :3]
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = c2w1[:3, 3] + t * (c2w2[:3, 3] - c2w1[:3, 3])
        return c2w

class VideoSequence:
    def __init__(self, camera_set, camera_idx_set, cam_params, seq_id):
        self.camera_set = camera_set
        self.camera_idx_set = camera_idx_set
        self.cam_params = cam_params
        self.train_cam_idx = None
        self.closest_idx = None 
        self.seq_id = seq_id
        self.set_train_cam_idx()
        self.debug()
        
    def debug(self):
        for cam, idx in zip(self.camera_set, self.camera_idx_set):
            print("[DEBUG] : VideoSequence : ", self.seq_id, cam.image_name, idx, cam.ref_img_name)
        
    def set_train_cam_idx(self):
        self.debug()
        
        ref_img_name = self.camera_set[0].ref_img_name
        seq = self.camera_set
        print("[DEBUG] : self.seq_id = ", self.seq_id, "ref_img_name = ", ref_img_name)
        # WARNING : hardcoded for LLFF-Extra and Nerfbusters
        # train_cam_idx = int(ref_img_name.replace("frame_", "")) - 1 if "frame_" in ref_img_name else int(ref_img_name)
        if seq[0].traj_id != None: # From VideoSequence
            if self.cam_params.search_search_strategy != "freespace-sampling":
                closest_idx = 0
            else:
                ts = [float(cam.image_name.split("_t")[-1]) for cam in seq]
                closest_idx = int(np.argmin(np.abs(1 - np.array(ts))))
        else:
            if self.cam_params.sampling_strategy == "src-test-ref-train":
                ts = [float(cam.image_name.split("_t")[-1]) for cam in seq]
                closest_idx = int(np.argmin(np.abs(1 - np.array(ts))))
            elif "sphere" in self.cam_params.sampling_strategy:
                costs = [cam.cost for cam in seq]
                closest_idx = int(np.argmin(costs))
            else:
                raise ValueError(f"[ERROR] : VideoSequence ; invalid sampling strategy : {self.cam_params.sampling_strategy}")
        # self.train_cam_idx = train_cam_idx
        self.ref_img_name = ref_img_name
        self.closest_idx = closest_idx
    
    def __len__(self):
        return len(self.camera_set)
    
    def __getitem__(self, idx):
        return self.camera_set[idx], self.camera_idx_set[idx]

    @property
    def cost(self):
        return sum([cam.cost for cam in self.camera_set if cam.virtual])

    @property
    def last_iter(self):
        last_iter_set = [cam.last_iter for cam in self.camera_set if cam.virtual]
        return sum(last_iter_set) / len(last_iter_set)

def quaternion_from_matrix(matrix, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Args:
        matrix: rotation matrix to obtain quaternion
        isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
        K = np.array(K)
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(
    quat0, quat1, fraction: float, spin: int = 0, shortestpath: bool = True
) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def quaternion_matrix(quaternion) -> np.array:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def unit_vector(data, axis = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    # np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data

def selection_high_priority_virtual_cams(cam_list, n_updates):
    """
    1st priority: not generated yet
    2nd priority: old updated order; sample 2 x num_selection
    3rd priority: low cost order; sample num_selection
    """
    selected_idxs = []
    not_gen_idxs = []  # high priority
    gen_idxs = []
    
    for idx, cam in enumerate(cam_list):
        if cam.original_image is None:
            not_gen_idxs.append(idx)
        else:
            gen_idxs.append(idx)
    
    # first filter
    candidates1_idxs = []
    if len(not_gen_idxs) > 0:
        candidates1_idxs += not_gen_idxs
    else:
        candidates1_idxs = list(range(len(cam_list)))  # all indices if none ungenerated
    
    # print("[DEBUG] : ", len(candidates1_idxs))
    
    if len(candidates1_idxs) > n_updates:
        # apply 2nd filter: based on last updated iteration
        costs = [cam_list[idx].cost for idx in candidates1_idxs]
        # print(costs)
        low_costs_idx = list(np.argsort(np.array(costs)))[:n_updates]
        return [candidates1_idxs[idx] for idx in low_costs_idx]
    else:
        extras_idxs = gen_idxs
        second_samples = 2 * (n_updates - len(candidates1_idxs))
        
        last_updated = [cam_list[idx].last_iter for idx in extras_idxs]
        old_updated_idx = list(np.argsort(np.array(last_updated)))[:second_samples]
        candidates2_idxs = [extras_idxs[idx] for idx in old_updated_idx]
        
        costs = [cam_list[idx].cost for idx in candidates2_idxs]
        low_costs_idx = list(np.argsort(np.array(costs)))[:n_updates - len(candidates1_idxs)]
        
        return candidates1_idxs + [candidates2_idxs[idx] for idx in low_costs_idx]

def export_cameras(cameras, save_path):
    import json
    json_cams = []
    for id, cam in enumerate(cameras):
        json_cams.append(camera_to_JSON(id, cam))
    with open(save_path, 'w') as file:
        json.dump(json_cams, file, indent=4)
    print("[INFO] : export cameras at ", save_path)


def save_camera_metadata(camera: Camera, save_path: str):
    """
    Save the metadata of a Camera instance into a JSON file,
    excluding the original_image and invdepthmap (or any other large tensors you don't want saved).
    """

    metadata = {}

    # Basic scalar/string fields
    metadata["uid"] = camera.uid
    metadata["colmap_id"] = camera.colmap_id
    metadata["FoVx"] = float(camera.FoVx)  # Convert to float
    metadata["FoVy"] = float(camera.FoVy)  # Convert to float
    metadata["image_name"] = camera.image_name
    metadata["use_view_direction"] = camera.use_view_direction
    metadata["data_device"] = str(camera.data_device)  # store device as a string
    metadata["image_width"] = int(camera.image_width)  # Convert to int
    metadata["image_height"] = int(camera.image_height)  # Convert to int
    metadata["depth_reliable"] = camera.depth_reliable
    metadata["znear"] = float(camera.znear)  # Convert to float
    metadata["zfar"] = float(camera.zfar)  # Convert to float
    metadata["trans"] = camera.trans.tolist() if isinstance(camera.trans, np.ndarray) else list(camera.trans)
    metadata["scale"] = float(camera.scale)  # Convert to float
    metadata["last_iter"] = int(camera.last_iter)  # Convert to int
    metadata["virtual"] = camera.virtual
    metadata["cost"] = float(camera.cost)  # Convert to float
    metadata["fl_expand"] = float(camera.fl_expand)  # Convert to float
    metadata["adjust_block_size"] = camera.adjust_block_size
    metadata["replace_img"] = camera.replace_img
    metadata["ref_img_name"] = camera.ref_img_name
    
    # Matrices/tensors that you do want to save
    # Note: we move them to CPU in case they're on GPU
    metadata["R"] = camera.R.tolist()  # Assuming R is a numpy array
    metadata["T"] = camera.T.tolist()  # Assuming T is a numpy array

    # The transforms, if you want to re-apply them exactly on load
    metadata["world_view_transform"] = camera.world_view_transform.cpu().numpy().tolist()
    metadata["projection_matrix"] = camera.projection_matrix.cpu().numpy().tolist()
    metadata["full_proj_transform"] = camera.full_proj_transform.cpu().numpy().tolist()
    metadata["camera_center"] = camera.camera_center.cpu().numpy().tolist()
    metadata["c2w"] = camera.c2w.cpu().numpy().tolist()

    # Exclude large data fields: original_image, invdepthmap, etc.
    # No action here, just making it explicit that we do NOT add them to `metadata`.

    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Camera metadata saved to {save_path.split('/')[-1]}")

def load_camera_metadata(load_path: str,
                        image: torch.Tensor = None,
                        extra_image: torch.Tensor = None,
                        invdepthmap=None,
                        depth_params=None,
                        data_device: str = "cuda"
                        ) -> Camera:
    """
    Load the metadata from JSON, then create a new Camera object. 
    Provide `image` (and others as needed) explicitly, since it wasn't stored in the metadata.
    """
    with open(load_path, "r") as f:
        metadata = json.load(f)

    # Convert JSON lists back to torch Tensors/numpy arrays
    R = np.asarray(metadata["R"], dtype=np.float32)
    T = np.asarray(metadata["T"], dtype=np.float32)

    # If you need them on CPU, you can place them on CPU, etc.
    world_view_transform = torch.tensor(metadata["world_view_transform"], dtype=torch.float32, device=data_device)
    projection_matrix = torch.tensor(metadata["projection_matrix"], dtype=torch.float32, device=data_device)
    full_proj_transform = torch.tensor(metadata["full_proj_transform"], dtype=torch.float32, device=data_device)
    camera_center = torch.tensor(metadata["camera_center"], dtype=torch.float32, device=data_device)
    c2w = torch.tensor(metadata["c2w"], dtype=torch.float32, device="cpu")

    # Rebuild the camera
    camera = Camera(
        colmap_id=metadata["colmap_id"],
        R=R,
        T=T,
        FoVx=metadata["FoVx"],
        FoVy=metadata["FoVy"],
        image=(image if image is not None else torch.zeros((3, metadata["image_height"], metadata["image_width"]))),
        image_name=metadata["image_name"],
        uid=metadata["uid"],
        trans=np.array(metadata["trans"]),
        scale=metadata["scale"],
        data_device=metadata["data_device"],  # or override with data_device
        use_view_direction=metadata["use_view_direction"],
        override_width=metadata["image_width"],
        override_height=metadata["image_height"],
        gt_alpha_mask=None,  # if you have a mask, set it here
        extra_image=extra_image,
        invdepthmap=invdepthmap.float() if invdepthmap is not None else None,
        depth_params=depth_params,
        cost=metadata["cost"]
    )

    # Now restore the attributes that were saved but not part of the constructor arguments
    camera.depth_reliable = metadata["depth_reliable"]
    camera.znear = metadata["znear"]
    camera.zfar = metadata["zfar"]
    camera.last_iter = metadata["last_iter"]
    camera.virtual = metadata["virtual"]
    camera.fl_expand = metadata["fl_expand"]
    camera.adjust_block_size = metadata["adjust_block_size"]
    camera.replace_img = metadata["replace_img"]
    camera.ref_img_name = metadata["ref_img_name"]

    # These transforms are typically computed inside the constructor,
    # but if you want to exactly match the saved values, you can replace them after initialization:
    camera.world_view_transform = world_view_transform
    camera.projection_matrix = projection_matrix
    camera.full_proj_transform = full_proj_transform
    camera.camera_center = camera_center
    camera.c2w = c2w

    print(f"Camera metadata loaded from {load_path.split('/')[-1]}")
    return camera