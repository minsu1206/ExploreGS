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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.space_search import SpaceSearch
from arguments import ModelParams, MaskingParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, mask_params : MaskingParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # if load_iteration:
        if load_iteration == -1:
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        else:
            self.loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # Original
        # if os.path.exists(os.path.join(args.source_path, "sparse")):
        #     scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, random_init=args.random_init)
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     assert False, "Could not recognize scene type!"
        kwargs = {"random_init":args.random_init} if hasattr(args, "random_init") else {}
        kwargs["eval"] = True
        kwargs["mode"] = args.dataset_distribution if hasattr(args, "dataset_distribution") else "default"
        if hasattr(args, "depths"):
            if args.depths != "":
                kwargs["depths"] = args.depths
        if hasattr(args, "oracle"):
            kwargs["oracle"] = args.oracle
            print("[INFO] : Oracle mode = ", args.oracle)
        
        if "scannetpp" in args.source_path:
            del kwargs["mode"]
        if "explore" in args.source_path:
            kwargs["mode"] = "train" if kwargs["mode"] == "easy" or kwargs["mode"] == "small-baseline" else kwargs["mode"] # default setting is easy. so change it to default value of WildExplore
        scene_type = args.loader if hasattr(args, "loader") else "Colmap"
        if "depths" in kwargs:
            scene_type += "-with-depth"
        data_read_func = sceneLoadTypeCallbacks[scene_type]
        # if scene_type == "Nerfbusters":
        #     del kwargs["mode"]
        scene_info = data_read_func(args.source_path, args.images, **kwargs)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=4)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_min_diff = scene_info.nerf_normalization["cameras_min_diff"]
        
        print("resolution_scales : ", resolution_scales)
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, mask_params)
            # if args.debug_w_trainset:
            #     self.train_cameras[resolution_scale] = [sorted(self.train_cameras[resolution_scale], key=lambda cam: cam.image_name)[0],
            #                                             sorted(self.train_cameras[resolution_scale], key=lambda cam: cam.image_name)[-1]]
            #     # self.train_cameras[resolution_scale] = sorted(self.train_cameras[resolution_scale], key=lambda cam: cam.image_name)[::2]
            #     for train_cam in self.train_cameras[resolution_scale]:
            #         print(train_cam.image_name) 

            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, mask_params)
            # if args.debug_w_trainset:
            #     self.test_cameras[resolution_scale] = [sorted(self.test_cameras[resolution_scale], key=lambda cam: cam.image_name)[0]]
                # self.test_cameras[resolution_scale] = sorted(self.test_cameras[resolution_scale], key=lambda cam: cam.image_name)[::5][:3]
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, mask_params)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, cam_infos=scene_info.train_cameras)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        if hasattr(self.gaussians, "exposure_mapping"):
            exposure_dict = {
                image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                for image_name in self.gaussians.exposure_mapping
            }
            with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
                json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]