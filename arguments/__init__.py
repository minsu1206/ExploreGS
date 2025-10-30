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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    def __repr__(self):
        # print all property
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

    def __repr__(self):
        # print all property
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.loader = "Colmap" # [Colmap, mipnerf360-sparse, mipnerf360-extrapolate, nerfbusters] # mipnerf360-extrapolate should deal with easy set / hard set
        self.random_init = False
        self.dataset_distribution = "easy"
        self.isotropic = False
        self.oracle = False
        # self.sparse = False
        # self.extrapolate = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False  # This makes very slow rasterizer ; from original option
        # debugging options
        self.export_camera = True
        self.freeze_gs = False
        self.test_set_type = "test" # [None, train, test, test-#]
        self.exit_update_all_once = False
        self.inspect_vcam = True
        self.antialiasing = False # from upgraded-3DGS
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05 
        self.min_opacity = 0.005
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.update_sh_interval = 1_000
        
        self.lambda_depth_dist = 0.0
        self.lambda_depth_gap = 0.0
        # from updated 3DGS
        self.inv_depth = True # 0117 : True is new default setting ; always use render_upgrade
        self.depth_l1_weight_init = 0.0
        self.depth_l1_weight_final = 0.0
        # also apply depth at stage2 (generated image)?
        self.depth_reg_gen = False
        # appearance modeling
        self.appearance_model = "" # ["", "expmaps"] # expmaps - from upgraded-3DGS (featsplat/hier not implemented)
        self.exposure_lr_init = 0.01 # for "expmaps"
        self.exposure_lr_final = 0.001 # for "expmaps"
        self.exposure_lr_delay_mult = 0.0 # for "expmaps"
        self.exposure_lr_delay_steps = 0 # for "expmaps"
        
        self.virtual_loss = "" # [dinov2, patchLPIPS]

        # confidence-aware
        self.pixel_confidence = False
        self.image_confidence = ""
        self.conf_once = False
        # guided sampling
        self.guided_sampling = False
        
        super().__init__(parser, "Optimization Parameters")


# TODO:
class DiffusionParams(ParamGroup):
    """
    Related to Diffusion Prior
    """
    def __init__(self, parser):
        self.prior_type = "enhancer" # [enhancer]
        self.input_resolution = 256

        self.model_root = ""
        self.model_ckpt_path = "" # trained model path
        self.model_config_path = "" # trained model config path
        self.num_steps = 50 # default value

        # mvinpaint params
        self.minibatch = -1 # N refer , M tgt
        self.n_tgt_views = 1
        self.crop = "center" # [center / random] > no-use // [squares]

        self.camera_normalization = True
        self.camera_relative_pose = True

        self.uncertainty_to_mask = "binarize"
        self.binarized_threshold = 3.0
        self.wrap_seq = False # for CameraGenerator

        super().__init__(parser, "Diffusion prior Paramters")

class VirtualCamParams(ParamGroup):
    """
    Related to virtual camera sampling & dataset update
    """
    def __init__(self, parser):
        self.sampling_strategy = "src-test-ref-train" # [src-test-ref-train (default), sphere-expansion]
        # pipeline
        self.dataset_update_iter = 100
        self.dataset_update_counts = 2
        self.dataset_update_percentage = -1 # if 1 >= val > 0, this override dataset_update_counts as percentage * N_virtual_views
        self.reference_strategy = "k-nearest" # [k-nearest random next-frame interpolate]
        # TODO: consider view-direction also

        # for automatic trajectory search
        self.search_bbox = "bbox" # [sphere / bbox / freespace-obb]
        self.search_voxel_resolution = -1 # -1 : disable
        self.search_coarse_voxel_resolution = -1 # -1 : disable
        self.search_search_strategy = ""
        self.search_connect_strategy = "simple-dist"
        self.search_mesh_dist_threshold = "1.0"
        self.search_use_frontier = False
        self.search_filter_params = [0,1,2,3,5] # 
        self.search_fisher_trick = ""
        self.search_space_sampling_strategy = "sphere" # [sphere (default) / look-center]
        self.search_travel_stop = ["length-16"] # length-N , path-K 
        self.search_anchor_set = ["6t", "4r"] # element : 6t , 4r, 2t
        self.search_orbit_threshold = False
        self.search_anchor_set_angle = 15 # degree
        self.search_anchor_set_angle_orbit = 15 # degree
        self.search_expand_dfs = True
        self.search_expand_strategy = [] # anchor / px / px_bg
        self.search_px_downsample = -1
        self.search_grid_no_use_occ = False
        
        """ search_expand_strategy's element : 
        - nbvs (use anchor set as action policy)
        - px (go to high uncertain region from px level uncertainty map ; exclude BG)
        - px_bg (go to high uncertain region from px level uncertainty map ; include BG)
        """
        self.search_topk = 1
        self.search_aux_bbox_ratio = 2.0
        self.search_early_guard = False
        self.search_subsample = -1
        self.search_bg_reject_ratio = 0.5 # default
        self.search_fix_right_vec = False
        self.search_fix_up_vec = False
        self.search_dir_angle = 30 # default
        self.search_space_guard = "" # [free / frontier]
        self.search_gs_proj = False
        self.search_gain_weight = ""
        self.search_travel_control = "greedy"    # [greedy / all / drop-half]
        self.search_mesh_leave_largest = 50
        self.search_occ_threshold = 0.5 # default
        self.search_reject_depth = "mix_1.0_1.0"
        self.search_mesh_res = 256
        self.search_skip_rejection = False
        self.search_skip_freespace_rejection = False
        
        # diffusion
        self.n_ref_views = 10
        self.search_reverse_top = False # for ablation?
        
        super().__init__(parser, "Virtual Camera Sampling Paramters")

class MaskingParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        ## Ours (1) uncertainty-vmf
        # from ModelParams
        self.use_view_direction = False
        self.use_visibility = False # CUSTOM
        self.use_visibility_mask = False # from ExtraNeRF
        self.use_scale_mask = False # from 3DGS-Enhancer
        self.use_viewdirection_mask = False

        # from OptimizationParams
        self.visibility_weight = False
        self.use_fisher = False
        self.fisher_current = False
        
        self.filter = [1,2]
        self.pruning = -1
        self.replace_gt = False
        self.use_ref = False
        self.nodepth = False
        
        super().__init__(parser, "Uncertainty-Mask-related Paramters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    print(cmdlne_string, args_cmdline)
    
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
