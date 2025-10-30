import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
import os
from read_write_model import read_model

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images_metas[key].point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus")
    # parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument('--scene', default="aloe")
    parser.add_argument('--model_type', default="bin")
    parser.add_argument('--dataset', default="nerfbusters-dataset")
    args = parser.parse_args()

    # args.base_dir = f"/workspace/dataset/nerfbusters-dataset/{args.scene}"
    # args.depths_dir = f"/workspace/dataset/nerfbusters-dataset/{args.scene}/depths"

    args.base_dir = f"/workspace/dataset/{args.dataset}/{args.scene}"
    if args.dataset == "scannetpp_dslr_only":
        args.base_dir = os.path.join(args.base_dir, "dslr")
        args.model_type = "txt"
    args.depths_dir = f"{args.base_dir}/depths"

    print("[DEBUG] : args.base_dir : ", args.base_dir, os.path.exists(args.base_dir))
    print("[DEBUG] : args.depths_dir : ", args.depths_dir, os.path.exists(args.depths_dir))
    print("[DEBUG] : colmap : ", os.path.exists(os.path.join(args.base_dir, "colmap")), args.dataset == "scannetpp_dslr_only")

    if os.path.exists(os.path.join(args.base_dir, "sparse", "0")):
        cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")
    elif os.path.exists(os.path.join(args.base_dir, "colmap/sparse/0")):
        cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "colmap/sparse/0"), ext=f".{args.model_type}")
    elif os.path.exists(os.path.join(args.base_dir, "colmap")) and args.dataset == "scannetpp_dslr_only":
        cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "colmap"), ext=f".{args.model_type}")
    
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    if os.path.exists(os.path.join(args.base_dir, "sparse", "0")):
        with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
            json.dump(depth_params, f, indent=2)
    else:
        
        if args.dataset == "nerfbusters-dataset":
            with open(f"{args.base_dir}/model_train/sparse/depth_params.json", "w") as f:
                json.dump(depth_params, f, indent=2)
        elif args.dataset == "scannetpp_dslr_only":
            with open(f"{args.base_dir}/colmap/depth_params.json", "w") as f:
                json.dump(depth_params, f, indent=2)
        else:
            with open(f"{args.base_dir}/depth_params.json", "w") as f:
                json.dump(depth_params, f, indent=2)

    print(0)