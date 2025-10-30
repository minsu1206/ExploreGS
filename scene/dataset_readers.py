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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_params: dict
    depth_path: str

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    def compute_min_difference(cam_centers):
        min_difference = np.inf
        n = len(cam_centers)
        for i in range(n):
            for j in range(i + 1, n):
                difference = np.linalg.norm(cam_centers[i] - cam_centers[j])
                min_difference = min(min_difference, difference)
        return min_difference

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius, 
            "cameras_min_diff":compute_min_difference(cam_centers)}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in tqdm(enumerate(cam_extrinsics), desc="Reading ColamCameras"):
        # sys.stdout.write('\r')
        # the exact output you're looking for:
        # sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV": # for nerfbusters
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, cy * 2)
            FovX = focal2fov(focal_length_x, cx * 2)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # R: c2w ; T: c2w
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth_params=None, depth_path="")
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras_with_depth(cam_extrinsics, cam_intrinsics, images_folder, depth_folder, depths_params):
    
    cam_infos = []
    for idx, key in tqdm(enumerate(cam_extrinsics), desc="Reading ColamCameras with Depth"):
        # sys.stdout.write('\r')
        # the exact output you're looking for:
        # sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV": # for nerfbusters
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, cy * 2)
            FovX = focal2fov(focal_length_x, cx * 2)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        # print(f"[INFO] : depths_params : {depths_params}")
        
        if depths_params is not None:
            # print("[INFO] : extr.name[:-n_remove] : ", extr.name[:-n_remove])
            
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        depth_path = os.path.join(depth_folder, f"{extr.name[:-n_remove]}.png") if depth_folder != "" else ""
        
        # R: c2w ; T: c2w
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, random_init=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        partition_path = os.path.join(path, "partition.json")
        if os.path.exists(partition_path): # For LLFF-extra
            with open(partition_path, "r") as f:
                partition = json.load(f)
            train_idx_set = partition["train"]
            test_idx_set = partition["test"]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx_set]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx_set]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path) and not random_init:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    if random_init:
        # num_pts = 100_000 # default : 100k
        num_pts = 1000 # 1k
        print(f"Random initialize {num_pts} points ; Dense Sparse Variation following Rain-GS")
        cam_pos = []
        for k in cam_extrinsics.keys():
            cam_pos.append(cam_extrinsics[k].tvec)
        cam_pos = np.array(cam_pos)
        min_cam_pos = np.min(cam_pos)
        max_cam_pos = np.max(cam_pos)
        mean_cam_pos = (min_cam_pos + max_cam_pos) / 2.0
        cube_mean = (max_cam_pos - min_cam_pos) * 1.5
        xyz = np.random.random((num_pts, 3))
        xyz = xyz + nerf_normalization["translate"]
        
        shs = np.random.random((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=shs, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None  # pcd=None makes error later.

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfbustersInfo(path, images, eval=True, downsample_factor=2, random_init=False, mode="default", oracle=False):
    # may be necessary but not now.
    # nerfbusters_scenes = ["aloe", "art", "car", "century", "flowers", "garbage", "picnic", \
    #                     "pikachiu", "pipe", "plant", "roses", "roses", "table"]
    # for scene_name in nerfbusters_scenes:
    #     if scene_name in path: break
    if mode == "swap":
        print("[INFO] : readNerfbustersInfo : mode = swap ; train/test split is changed")

    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    # TODO: resolution types
    scene_name = Path(path).stem
    if scene_name in ["century"]: # table : original = 1080p
        reading_dir = f"images"
    else:
        reading_dir = f"images_{downsample_factor}"
    # reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(
                        cam_extrinsics=cam_extrinsics, 
                        cam_intrinsics=cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = []
    test_cam_infos = []
    for idx, c in enumerate(cam_infos):
        if "frame_1_" in c.image_name:
            if mode == "swap":
                train_cam_infos.append(c)
            else:
                test_cam_infos.append(c)
        else:
            if mode == "swap":
                test_cam_infos.append(c)
            else:
                train_cam_infos.append(c)
    if oracle:
        train_cam_infos.extend(test_cam_infos) # include test cameras for training
    print(f"Camera infos : \n # cameras for train = {len(train_cam_infos)} \n # cameras for eval = {len(test_cam_infos)}")
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    if mode == "swap":
        bin_path = os.path.join(path, "model_test/sparse/points3D.bin")
    else:
        bin_path = os.path.join(path, "model_train/sparse/points3D.bin")
    ply_path = bin_path.replace(".bin", ".ply")
    xyz, rgb, _ = read_points3D_binary(bin_path)
    storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfbustersInfo_with_depth(path, images, depths, eval=True, downsample_factor=2, random_init=False, mode="default", oracle=False):
    print(f"[INFO] : readNerfbustersInfo_with_depth")
    
    if mode == "swap":
        print("[INFO] : readNerfbustersInfo : mode = swap ; train/test split is changed")

    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    colmap_dir = "model_test/sparse" if mode == "swap" else "model_train/sparse"
    depth_params_file = os.path.join(path, "model_train/sparse", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
    
    # TODO: resolution types
    scene_name = Path(path).stem
    if scene_name in ["century"]:
        reading_dir = f"images"
    else:
        reading_dir = f"images_{downsample_factor}"
    # reading_dir = "images"
    cam_infos_unsorted = readColmapCameras_with_depth(
                        cam_extrinsics=cam_extrinsics, 
                        cam_intrinsics=cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir),
                        depth_folder=os.path.join(path, depths),
                        depths_params=depths_params)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = []
    test_cam_infos = []
    for idx, c in enumerate(cam_infos):
        if "frame_1_" in c.image_name:
            if mode == "swap":
                train_cam_infos.append(c)
            else:
                test_cam_infos.append(c)
        else:
            if mode == "swap":
                test_cam_infos.append(c)
            else:
                train_cam_infos.append(c)
    if oracle:
        train_cam_infos.extend(test_cam_infos) # include test cameras for training
    print(f"Camera infos : \n # cameras for train = {len(train_cam_infos)} \n # cameras for eval = {len(test_cam_infos)}")
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    if mode == "swap":
        bin_path = os.path.join(path, "model_test/sparse/points3D.bin")
    else:
        bin_path = os.path.join(path, "model_train/sparse/points3D.bin")
    ply_path = bin_path.replace(".bin", ".ply")
    xyz, rgb, _ = read_points3D_binary(bin_path)
    storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readWildExploreInfo(path, images, eval=True, downsample_factor=2, random_init=False, mode="train", oracle=False):
    """
    mode : [train explore1 explore2]
    """
    print("[INFO] : readWildExploreInfo ; mode = ", mode)
    
    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "model_train/sparse/0/images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "model_train/sparse/0/cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "model_train/images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "model_train/cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    index_info_json = os.path.join(path, "colmap/sparse/0", "3dgs_info.json")
    with open(index_info_json, "r") as f:
        index_info = json.load(f)
    
    # TODO: resolution types
    scene_name = Path(path).stem
    reading_dir = "images_2"
    # reading_dir = "images"
    if not os.path.exists(os.path.join(path, reading_dir)):
        raise FileNotFoundError(f"[ERROR] : {reading_dir} not found at path '{path}'.")
    
    cam_infos_unsorted = readColmapCameras(
                        cam_extrinsics=cam_extrinsics, 
                        cam_intrinsics=cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos_raw = []
    test_cam_infos = []
    for idx, c in enumerate(cam_infos):
        if "train" in c.image_name:
            train_cam_infos_raw.append(c)
        # else:
        #     test_cam_infos.append(c)
        # if mode == "train":
        #     if "train" in c.image_name:
        #         train_cam_infos_raw.append(c)
        #     else: # include 
        #         test_cam_infos.append(c)
        # elif mode == "interpolation" or mode == "extrapolation":
        #     if "train" in c.image_name:
        #         train_cam_infos_raw.append(c)
        # else:
        #     pass
    print("[DEBUG] : train_cam_infos_raw : ", len(train_cam_infos_raw))
    print("[DEBUG] : index_info : ", index_info)
    
    train_cam_indices = list(set(index_info["train"]) - set(index_info["interpolation"]) - set(index_info["extrapolation"]))
    print("[DEBUG] : train_cam_indices : ", train_cam_indices)
    train_cam_infos = [train_cam_infos_raw[i] for i in train_cam_indices]
    try:
        interp_cam_infos = [train_cam_infos_raw[i] for i in index_info["interpolation"]]
    except: # index out
        interp_cam_infos = []
    try:
        extrap_cam_infos = [train_cam_infos_raw[i] for i in index_info["extrapolation"]]
    except: # index out
        extrap_cam_infos = []
    print("[DEBUG] : interp : ", [c.image_name for c in interp_cam_infos])
    print("[DEBUG] : extrap : ", [c.image_name for c in extrap_cam_infos])
    
    
    
    explore1_path = os.path.join(path, "colmap/sparse/0", "model_explore1")
    explore2_path = os.path.join(path, "colmap/sparse/0", "model_explore2")
    
    if os.path.exists(explore1_path):
        explore1_cam_intrinsics = read_intrinsics_text(os.path.join(explore1_path, "sparse/0/cameras.txt"))
        explore1_cam_extrinsics = read_extrinsics_text(os.path.join(explore1_path, "sparse/0/images.txt"))
        
        explore1_cam_infos = readColmapCameras(
                        cam_extrinsics=explore1_cam_extrinsics, 
                        cam_intrinsics=explore1_cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir))
        explore1_cam_infos = sorted(explore1_cam_infos.copy(), key = lambda x : x.image_name)
        
        if mode == "train" or mode == "explore1":
            test_cam_infos.extend(explore1_cam_infos)
        
    if os.path.exists(explore2_path):
        explore2_cam_intrinsics = read_intrinsics_text(os.path.join(explore2_path, "sparse/0/cameras.txt"))
        explore2_cam_extrinsics = read_extrinsics_text(os.path.join(explore2_path, "sparse/0/images.txt"))
        
        explore2_cam_infos = readColmapCameras(
                        cam_extrinsics=explore2_cam_extrinsics, 
                        cam_intrinsics=explore2_cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir))
        explore2_cam_infos = sorted(explore2_cam_infos.copy(), key = lambda x : x.image_name)
        
        if mode == "train" or mode == "explore2":
            test_cam_infos.extend(explore2_cam_infos)
    # redundant
    if mode == "interpolation":
        test_cam_infos.extend(interp_cam_infos)
    elif mode == "extrapolation":
        test_cam_infos.extend(extrap_cam_infos)
    
    if oracle:
        train_cam_infos.extend(test_cam_infos) # include test cameras for training
    
    print(f"Camera infos : \n # cameras for train = {len(train_cam_infos)} \n # cameras for eval = {len(test_cam_infos)}")
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    bin_path = os.path.join(path, "colmap/sparse/0/model_train/sparse/0/points3D.bin")
    ply_path = bin_path.replace(".bin", ".ply")
    xyz, rgb, _ = read_points3D_binary(bin_path)
    storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readWildExploreInfo_with_depth(path, images, depths, eval=True, downsample_factor=2, random_init=False, mode="default", oracle=False):
    print(f"[INFO] : readWildExploreInfo_with_depth ; mode = ", mode)
    
    if mode == "swap":
        print("[INFO] : readNerfbustersInfo : mode = swap ; train/test split is changed")

    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    index_info_json = os.path.join(path, "colmap/sparse/0", "3dgs_info.json")
    with open(index_info_json, "r") as f:
        index_info = json.load(f)
    
    depth_params_file = os.path.join(path, "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
    
    # TODO: resolution types
    scene_name = Path(path).stem
    reading_dir = "images_2"
    if not os.path.exists(os.path.join(path, reading_dir)):
        raise FileNotFoundError(f"[ERROR] : {reading_dir} not found at path '{path}'.")
    
    cam_infos_unsorted = readColmapCameras_with_depth(
                        cam_extrinsics=cam_extrinsics, 
                        cam_intrinsics=cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir),
                        depth_folder=os.path.join(path, depths),
                        depths_params=depths_params)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos_raw = []
    test_cam_infos = []
    for idx, c in enumerate(cam_infos):
        if mode == "train":
            if "train" in c.image_name:
                train_cam_infos_raw.append(c)
            else: # include 
                test_cam_infos.append(c)
        elif mode == "interpolation" or mode == "extrapolation":
            if idx in index_info["train"]:
                train_cam_infos_raw.append(c)
        else:
            if idx in index_info["train"]:
                train_cam_infos_raw.append(c)

    train_cam_indices = list(set(index_info["train"]) - set(index_info["interpolation"]) - set(index_info["extrapolation"]))
    train_cam_infos = [train_cam_infos_raw[i] for i in train_cam_indices]
    try:
        interp_cam_infos = [train_cam_infos_raw[i] for i in index_info["interpolation"]]
    except:
        interp_cam_infos = []
    try:
        extrap_cam_infos = [train_cam_infos_raw[i] for i in index_info["extrapolation"]]
    except:
        extrap_cam_infos = []
    print("[DEBUG] : interp : ", [c.image_name for c in interp_cam_infos])
    print("[DEBUG] : extrap : ", [c.image_name for c in extrap_cam_infos])
    
    explore1_path = os.path.join(path, "colmap/sparse/0", "model_explore1")
    explore2_path = os.path.join(path, "colmap/sparse/0", "model_explore2")
    
    # only load depth for train
    if os.path.exists(explore1_path):
        explore1_cam_intrinsics = read_intrinsics_text(os.path.join(explore1_path, "sparse/0/cameras.txt"))
        explore1_cam_extrinsics = read_extrinsics_text(os.path.join(explore1_path, "sparse/0/images.txt"))
        
        explore1_cam_infos = readColmapCameras(
                        cam_extrinsics=explore1_cam_extrinsics, 
                        cam_intrinsics=explore1_cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir))
        explore1_cam_infos = sorted(explore1_cam_infos.copy(), key = lambda x : x.image_name)
        
        if mode == "train" or mode == "explore1":
            test_cam_infos.extend(explore1_cam_infos)
        
    if os.path.exists(explore2_path):
        explore2_cam_intrinsics = read_intrinsics_text(os.path.join(explore2_path, "sparse/0/cameras.txt"))
        explore2_cam_extrinsics = read_extrinsics_text(os.path.join(explore2_path, "sparse/0/images.txt"))
        
        explore2_cam_infos = readColmapCameras(
                        cam_extrinsics=explore2_cam_extrinsics, 
                        cam_intrinsics=explore2_cam_intrinsics, 
                        images_folder=os.path.join(path, reading_dir))
        explore2_cam_infos = sorted(explore2_cam_infos.copy(), key = lambda x : x.image_name)
        
        if mode == "train" or mode == "explore2":
            test_cam_infos.extend(explore2_cam_infos)
    
    if mode == "interpolation":
        test_cam_infos.extend(interp_cam_infos)
    elif mode == "extrapolation":
        test_cam_infos.extend(extrap_cam_infos)
    
    if oracle:
        train_cam_infos.extend(test_cam_infos) # include test cameras for training
    
    print(f"Camera infos : \n # cameras for train = {len(train_cam_infos)} \n # cameras for eval = {len(test_cam_infos)}")
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    bin_path = os.path.join(path, "colmap/sparse/0/model_train/sparse/0/points3D.bin")
    ply_path = bin_path.replace(".bin", ".ply")
    xyz, rgb, _ = read_points3D_binary(bin_path)
    storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def process_frame_list(frames, dslr_path, width, height, fov_x, fov_y, downsample_factor):
    cam_infos = []
    for uid, frame in tqdm(enumerate(frames), total=len(frames), desc="ScanNet++ : Processing frames"):
        image_name = os.path.basename(frame["file_path"])
        img_path = os.path.join(dslr_path, "resized_undistorted_images", frame["file_path"])
        transform_matrix = np.array(frame["transform_matrix"])
        # Convert Nerfstudio to COLMAP convention
        transform_matrix[2, :] *= -1
        transform_matrix = transform_matrix[[1, 0, 2, 3], :]
        transform_matrix[0:3, 1:3] *= -1
        # cam2world
        R = transform_matrix[:3, :3]
        world2cam = np.linalg.inv(transform_matrix)
        T = world2cam[:3, 3] 
        
        if not os.path.exists(img_path):
            print("[WARNING] : image not found at ", img_path)
            continue
        
        image = Image.open(img_path).convert("RGB").resize((width, height))
        if downsample_factor != 1:
            image = image.resize((width // downsample_factor, height // downsample_factor))

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=fov_y,
            FovX=fov_x,
            image=image,
            image_path=img_path,
            image_name=image_name,
            width=width,
            height=height,
            depth_params=None,
            depth_path=""
        )
        cam_infos.append(cam_info)
    return cam_infos

def readScanNetPPInfo(path, images, eval=True, downsample_factor=2, random_init=False, oracle=False):
    
    dslr_path = os.path.join(path, "dslr")
    json_path = os.path.join(dslr_path, "nerfstudio/transforms_undistorted.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    height = data["h"]
    width = data["w"]
    fx = data["fl_x"]
    fy = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]
    fov_x = focal2fov(fx, width)
    fov_y = focal2fov(fy, height)
    
    train_infos = data["frames"]
    test_infos = data["test_frames"]
    
    process_frame_pkg = {
        "dslr_path": dslr_path,
        "width": width,
        "height": height,
        "downsample_factor": downsample_factor,
        "fov_x": fov_x,
        "fov_y": fov_y
    }
    
    train_cam_infos = process_frame_list(frames=sorted(train_infos, key=lambda x: x["file_path"]), **process_frame_pkg)
    test_cam_infos = process_frame_list(frames=sorted(test_infos, key=lambda x: x["file_path"]) if eval else [], **process_frame_pkg)

    if oracle:
        train_cam_infos.extend(test_cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(dslr_path, "colmap/points3D.txt")
    
    xyz, rgb, _ = read_points3D_text(ply_path)
    print("[INFO] : ScanNet++ : # of initial point cloud = ", len(xyz))
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((len(xyz), 3)))
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readScanNetPPInfo_with_depth(path, images, depths, eval=True, downsample_factor=2, random_init=False, oracle=False):
    dslr_path = os.path.join(path, "dslr")
    json_path = os.path.join(dslr_path, "nerfstudio/transforms_undistorted.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    height = data["h"]
    width = data["w"]
    fx = data["fl_x"]
    fy = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]
    fov_x = focal2fov(fx, width)
    fov_y = focal2fov(fy, height)
    
    train_infos = data["frames"]
    test_infos = data["test_frames"]
    
    process_frame_pkg = {
        "dslr_path": dslr_path,
        "width": width,
        "height": height,
        "downsample_factor": downsample_factor,
        "fov_x": fov_x,
        "fov_y": fov_y
    }
    train_cam_infos = process_frame_list(frames=sorted(train_infos, key=lambda x: x["file_path"]), **process_frame_pkg)
    test_cam_infos = process_frame_list(frames=sorted(test_infos, key=lambda x: x["file_path"]) if eval else [], **process_frame_pkg)

    if oracle:
        train_cam_infos.extend(test_cam_infos)
    
    # Load depth params
    depth_params = None
    depth_params_file = os.path.join(dslr_path, "colmap/depth_params.json")
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depth_params = json.load(f)
            all_scales = np.array([depth_params[key]["scale"] for key in depth_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depth_params:
                depth_params[key]["scale"] = med_scale
        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
    # Update depth params
    for cam_info in train_cam_infos:
        if cam_info.image_name in depth_params:
            cam_info.depth_params = depth_params[cam_info.image_name]
            cam_info.depth_path = os.path.join(dslr_path, "depths", cam_info.image_name)
    for cam_info in test_cam_infos:
        if cam_info.image_name in depth_params:
            cam_info.depth_params = depth_params[cam_info.image_name]
            cam_info.depth_path = os.path.join(dslr_path, "depths", cam_info.image_name)
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(dslr_path, "colmap/points3D.txt")
    
    xyz, rgb, _ = read_points3D_text(ply_path)
    print("[INFO] : ScanNet++ : # of initial point cloud = ", len(xyz))
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((len(xyz), 3)))
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Nerfbusters": readNerfbustersInfo,
    "Nerfbusters-with-depth": readNerfbustersInfo_with_depth,
    "WildExplore": readWildExploreInfo,
    "WildExplore-with-depth": readWildExploreInfo_with_depth,
    "ScanNetPP": readScanNetPPInfo,
    "ScanNetPP-with-depth": readScanNetPPInfo_with_depth,
}