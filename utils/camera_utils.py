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

from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, getWorld2View2
from typing import NamedTuple, Optional, List, Tuple
import copy
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, mask_params):
    orig_w, orig_h = cam_info.image.size

    if cam_info.depth_path != "":
        try:
            # if is_nerf_synthetic:
            #     invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            # else:
            invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
    
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    use_view_direction_ = args.use_view_direction if hasattr(args, "use_view_direction") else False
    use_view_direction_ = use_view_direction_ or (mask_params.use_view_direction if hasattr(mask_params, "use_view_direction") else False)
    
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  invdepthmap=invdepthmap, depth_params=cam_info.depth_params,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  use_view_direction=use_view_direction_)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, mask_params):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, mask_params))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    pos = C2W[:3, 3]
    rot = C2W[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    w = camera.width if hasattr(camera, "width") else camera.image_width
    h = camera.height if hasattr(camera, "height") else camera.image_height
    fovy = camera.FovY if hasattr(camera, "FovY") else camera.FoVy
    fovx = camera.FovX if hasattr(camera, "FovX") else camera.FoVx
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : w,
        'height' : h,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(fovy, h),
        'fx' : fov2focal(fovx, w)
    }
    return camera_entry

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def generate_hemispherical_orbit(poses, n_frames=120):
    """Calculates a render path which orbits around the z-axis."""
    origins = poses[:, :3, 3]
    radius = np.sqrt(np.mean(np.sum(origins**2, axis=-1)))

    # Assume that z-axis points up towards approximate camera hemisphere
    sin_phi = np.mean(origins[:, 2], axis=0) / radius
    cos_phi = np.sqrt(1 - sin_phi**2)
    render_poses = []

    up = np.array([0., 0., 1.])
    for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
        camorigin = radius * np.array(
            [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi])
        render_poses.append(viewmatrix(camorigin, up, camorigin))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses

def average_poses(poses: np.ndarray) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def create_cam2world_keep_origin(forward_vector, origin, world_up=None):
    forward_vector = forward_vector / np.linalg.norm(forward_vector, axis=1, keepdims=True)
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64).reshape(1, 3)
    else:
        world_up = world_up.reshape(1, 3)
    up_vector = np.tile(world_up, (forward_vector.shape[0], 1))
    
    right_vector = -np.cross(up_vector, forward_vector, axis=-1)
    right_vector = right_vector / np.linalg.norm(right_vector, axis=1, keepdims=True)
    
    up_vector = np.cross(forward_vector, right_vector, axis=-1)
    up_vector = up_vector / np.linalg.norm(up_vector, axis=-1, keepdims=True)
    
    rotation_matrix = np.tile(np.eye(4)[None], (forward_vector.shape[0], 1, 1))
    rotation_matrix[:, :3, :3] = np.stack([right_vector, up_vector, forward_vector], axis=-1)
    # rotation_matrix[:, :3, :3] = np.stack([right_vector, -forward_vector, up_vector], axis=-1)

    c2w = np.tile(np.eye(4)[None], (forward_vector.shape[0], 1,1))
    c2w[:, :3, :3] = rotation_matrix[:, :3, :3]
    # c2w[:, :3, :3] = np.tile(np.eye(3)[None], (forward_vector.shape[0], 1,1))
    c2w[:, :3,  3] = origin
    return c2w

def create_cam2world_matrix(forward_vector, origin, world_up=None):
    forward_vector = forward_vector / np.linalg.norm(forward_vector, axis=1, keepdims=True)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64).reshape(1, 3) if world_up is None else world_up.reshape(1,3)
    up_vector = np.tile(world_up, (forward_vector.shape[0], 1))
    
    right_vector = -np.cross(up_vector, forward_vector, axis=-1)
    right_vector = right_vector / np.linalg.norm(right_vector, axis=1, keepdims=True)
    
    up_vector = np.cross(forward_vector, right_vector, axis=-1)
    up_vector = up_vector / np.linalg.norm(up_vector, axis=-1, keepdims=True)
    
    rotation_matrix = np.tile(np.eye(4)[None], (forward_vector.shape[0], 1, 1))
    rotation_matrix[:, :3, :3] = np.stack([right_vector, up_vector, forward_vector], axis=-1)

    translation_matrix = np.tile(np.eye(4)[None], (forward_vector.shape[0], 1, 1))
    translation_matrix[:, :3, 3] = origin
    
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    # assert(cam2world.shape[1:] == (4, 4))
    return cam2world

def rotation_matrix_x(elevation):
    """
    Rotation matrix around the X-axis (horizontal axis).
    
    Args:
        elevation: The elevation angle in radians.
    
    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    cos_e = np.cos(elevation)
    sin_e = np.sin(elevation)
    return np.array([
        [1, 0, 0],
        [0, cos_e, -sin_e],
        [0, sin_e, cos_e]
    ])

def rotation_matrix_y(azimuth):
    """
    Rotation matrix around the Y-axis (vertical axis).
    
    Args:
        azimuth: The azimuth angle in radians.
    
    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    cos_a = np.cos(azimuth)
    sin_a = np.sin(azimuth)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
