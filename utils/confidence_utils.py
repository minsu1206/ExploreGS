import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import render_based_on_mask_type
from masking.visibility_mask import render_visibility_mask_bg_filter
from masking.scale_mask import render_scale_mask 
import os
import torchvision
import numpy as np
import contextlib

class ConfidenceScore:
    def __init__(self, args, opt, scene, pipe, mask_params, bg, render_func):
        self.args = args
        self.opt = opt
        self.scene = scene
        self.pipe = pipe
        self.mask_params = mask_params
        self.bg = bg
        self.render_func = render_func
        
        self.vcam_dbg_dir = os.path.join(args.model_path, "vcam", "debugging")
        os.makedirs(self.vcam_dbg_dir, exist_ok=True)
        
        # TODO: integrate DINO or LPIPS
        self.device = "cuda"
        
        if self.opt.pixel_confidence == "patch-lpips":
            import lpips
            self.px_conf_model = lpips.LPIPS(net='vgg').to(self.device)
            self.n_px_patch = 14 # HARDCODED
        
        elif self.opt.pixel_confidence == "px-lpips" or self.opt.virtual_loss == "lpips" or self.opt.pixel_confidence == "scale-pxlpips":
            import lpips
            self.px_conf_model2 = lpips.LPIPS(net='vgg',
                                             spatial=True).to(self.device)

        elif self.opt.pixel_confidence == "dinov2" or self.opt.virtual_loss == "dinov2":
            dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            dinov2_vitb14.eval()
            self.px_conf_model = dinov2_vitb14.to(self.device)
            self.n_px_patch = 14 # HARDCODED
    
    def get_confidence_dict(self, conf_dict_pkg):
        
        with torch.no_grad():
            confidence_dict = {}
            # unpacking
            viewpoint_cam = conf_dict_pkg["viewpoint_cam"]
            if self.opt.conf_once:
                if viewpoint_cam.px_conf is not None:
                    print("[DEBUG] : px_confidence loaded !", viewpoint_cam.image_name)
                    confidence_dict["px_confidence"] = viewpoint_cam.px_conf
                else:
                    print("[DEBUG] : px_confidence is None", viewpoint_cam.image_name)
            
            train_cameras = conf_dict_pkg["train_cameras"]
            gaussians = conf_dict_pkg["gaussians"]
            scene = conf_dict_pkg["scene"]
            pipe = conf_dict_pkg["pipe"]
            mask_params = conf_dict_pkg["mask_params"]
            bg = conf_dict_pkg["bg"]
            render_func = conf_dict_pkg["render_func"]
            diff_params = conf_dict_pkg["diff_params"]
            iteration = conf_dict_pkg["iteration"]
            gen = conf_dict_pkg["gen"]
            gen = gen.to(self.device)
            render = conf_dict_pkg["image"]
            
            if self.opt.pixel_confidence:
                
                if "px_confidence" not in confidence_dict:
                    
                    # if self.opt.pixel_confidence == "uncertainty":
                    #     pkg = render_based_on_mask_type(iteration, [viewpoint_cam], scene.getTrainCameras(), gaussians, scene, pipe, mask_params, bg, render_func, threshold=diff_params.binarized_threshold, save_mode=False)
                    #     confidence_dict["px_confidence"] = pkg["uncertainty_map"][0].detach()
                        
                    if self.opt.pixel_confidence == "visibility-mask" or self.opt.pixel_confidence == "visibility":
                        pkg = render_visibility_mask_bg_filter(scene.model_path, "virtual", iteration, [viewpoint_cam], [scene.getTrainCameras()], gaussians, render_func, pipe, bg, threshold=diff_params.binarized_threshold, save_mode=False)
                        if self.opt.pixel_confidence == "visibility-mask":
                            confidence_dict["px_confidence"] = pkg["binary_mask"][0]
                        elif self.opt.pixel_confidence == "visibility":
                            confidence_dict["px_confidence"] = pkg["uncertainty_map"][0]
                        
                    elif self.opt.pixel_confidence == "scale":
                        pkg = render_scale_mask(scene.model_path, "virtual", iteration, [viewpoint_cam], gaussians, render_func, pipe, bg, save_mode=True)
                        confidence_dict["px_confidence"] = pkg # (H,W)
                        
                    elif self.opt.pixel_confidence == "scale-pxlpips":
                        if len(render.shape) == 3:
                            render = render[None]
                        if len(gen.shape) == 3:
                            gen = gen[None]
                        scale_pkg = render_scale_mask(scene.model_path, "virtual", iteration, [viewpoint_cam], gaussians, render_func, pipe, bg, save_mode=True)
                        pxlpips_pkg = self.px_conf_model2(render, gen)[0, 0]
                        confidence_dict["px_confidence"] = scale_pkg * pxlpips_pkg # (H,W)
                        print("[DEBUG] : scale-pxlpips confidence computed !")
                        
                    elif self.opt.pixel_confidence == "patch-lpips":
                        
                        h, w = render.shape[-2:]
                        resize_h, resize_w = h // self.n_px_patch * self.n_px_patch, w // self.n_px_patch * self.n_px_patch
                        patch_h = h // self.n_px_patch
                        patch_w = w // self.n_px_patch
                        if len(render.shape) == 3:
                            render = render[None]
                        if len(gen.shape) == 3:
                            gen = gen[None]
                        
                        render = F.interpolate(render, size=(resize_h, resize_w), mode="bilinear")
                        gen = F.interpolate(gen, size=(resize_h, resize_w), mode="bilinear")
                        
                        lpips_map = torch.zeros((self.n_px_patch, self.n_px_patch), device=self.device)
                        for i in range(self.n_px_patch):
                            for j in range(self.n_px_patch):
                                patch_render = render[:, :, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                                patch_gen = gen[:, :, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                                patch_conf = self.px_conf_model(patch_render, patch_gen)
                                lpips_map[i, j] = patch_conf
                                
                        lpips_map = F.interpolate(lpips_map[None, None], size=(h, w), mode="bilinear")[0, 0]
                        confidence_dict["px_confidence"] = lpips_map
                    
                    elif self.opt.pixel_confidence == "px-lpips":
                        if len(render.shape) == 3:
                            render = render[None]
                        if len(gen.shape) == 3:
                            gen = gen[None]
                        confidence_dict["px_confidence"] = self.px_conf_model2(render, gen)[0, 0]
                        torchvision.utils.save_image(confidence_dict["px_confidence"], f"{self.vcam_dbg_dir}/{viewpoint_cam.image_name}_px_conf.png")
                    
                    else:
                        raise NotImplementedError(f"Pixel confidence type '{self.opt.pixel_confidence}' is not supported")
                    
                    if self.opt.conf_once:
                        viewpoint_cam.px_conf = confidence_dict["px_confidence"]
                        # save confidence_dict
                        torchvision.utils.save_image(confidence_dict["px_confidence"], f"{self.vcam_dbg_dir}/{viewpoint_cam.image_name}_px_conf.png")
                
        if self.opt.image_confidence != "":
            
            if self.opt.image_confidence == "iou" or self.opt.image_confidence == "inv-iou":
                iou_score = self.calc_img_confidence(viewpoint_cam=viewpoint_cam, train_cameras=train_cameras, gaussians=gaussians, scene=scene, pipe=pipe, mask_params=mask_params, bg=bg, render_func=render_func)
                if self.opt.image_confidence == "inv-iou":
                    confidence_dict["img_confidence"] = 1 - iou_score
                else:
                    confidence_dict["img_confidence"] = iou_score
                
            elif self.opt.image_confidence == "cost" or self.opt.image_confidence == "inv-cost" or self.opt.image_confidence == "cost2":
                if self.opt.image_confidence == "inv-cost":
                    confidence_dict["img_confidence"] = 1 - viewpoint_cam.cost
                else:
                    confidence_dict["img_confidence"] = viewpoint_cam.cost
            else:
                raise NotImplementedError(f"Image confidence type {self.opt.image_confidence} is not supported")

        return confidence_dict

    def get_loss_dict(self, conf_dict_pkg):
        train_cameras = conf_dict_pkg["train_cameras"]
        gaussians = conf_dict_pkg["gaussians"]
        scene = conf_dict_pkg["scene"]
        pipe = conf_dict_pkg["pipe"]
        mask_params = conf_dict_pkg["mask_params"]
        bg = conf_dict_pkg["bg"]
        render_func = conf_dict_pkg["render_func"]
        diff_params = conf_dict_pkg["diff_params"]
        iteration = conf_dict_pkg["iteration"]
        gen = conf_dict_pkg["gen"]
        gen = gen.to(self.device)
        render = conf_dict_pkg["image"]
        loss_dict = {}
        
        if self.opt.virtual_loss == "lpips": # scalar value . only for loss
            print("[DEBUG] : virtual loss is lpips")
            if len(render.shape) == 3:
                render = render[None]
            if len(gen.shape) == 3:
                gen = gen[None]
            
            lpips_map = self.px_conf_model2(render, gen)
            loss_dict["loss"] = lpips_map
          
        elif self.opt.virtual_loss == "dinov2":

            h, w = render.shape[-2:]
            resize_h, resize_w = h // self.n_px_patch * self.n_px_patch, w // self.n_px_patch * self.n_px_patch
            if len(render.shape) == 3:
                render = render[None]
            if len(gen.shape) == 3:
                gen = gen[None]
            
            render = F.interpolate(render, size=(resize_h, resize_w), mode="bilinear")
            gen = F.interpolate(gen, size=(resize_h, resize_w), mode="bilinear")
        
            features_render = self.px_conf_model.forward_features(render)
            features_gen = self.px_conf_model.forward_features(gen)
            
            render_patch = features_render["x_norm_patchtokens"]
            gen_patch = features_gen["x_norm_patchtokens"]
            
            cos_sim = F.cosine_similarity(render_patch, gen_patch, dim=2)[0] # (1, h//n_px_patch, w//n_px_patch)
            cos_sim = cos_sim.reshape(h // self.n_px_patch, w // self.n_px_patch)
            cos_sim = F.interpolate(cos_sim[None, None], size=(h, w), mode="bilinear")[0, 0]
            loss_dict["loss"] = cos_sim
        
        return loss_dict

    # TODO: implement
    @torch.no_grad()
    def calc_img_confidence(self, viewpoint_cam, train_cameras, gaussians, scene, pipe, mask_params, bg, render_func):
        # (1) compute nearest neighbor camera from train_cameras
        # (2) render image using nearest neighbor camera & viewpoint_cam
        # (3) compute confidence score
        query_origin = viewpoint_cam.c2w.numpy()[:3, 3]
        ref_origins = [cam.c2w.numpy()[:3, 3] for cam in train_cameras]
        ref_origin_distances = [np.linalg.norm(query_origin - ref_origin) for ref_origin in ref_origins]
        nearest_neighbor_idx = np.argmin(ref_origin_distances)
        nearest_neighbor_cam = train_cameras[nearest_neighbor_idx]

        # (2) render image using nearest neighbor camera & viewpoint_cam
        render_dict_nn = render_func(nearest_neighbor_cam, gaussians, pipe, bg)
        render_dict_viewpoint = render_func(viewpoint_cam, gaussians, pipe, bg)
        
        # import pdb; pdb.set_trace()
        
        nn_visible = render_dict_nn["visibility_filter"]
        viewpoint_visible = render_dict_viewpoint["visibility_filter"]
        # (3) compute confidence score : ID - IOU
        
        co_visible = nn_visible & viewpoint_visible
        once_visible = nn_visible | viewpoint_visible  # or operation
        iou = (co_visible).sum() / once_visible.sum()
        
        iou_score = iou.item()
        
        nn_render = render_dict_nn["render"]
        viewpoint_render = render_dict_viewpoint["render"]
        concat = torch.cat([nn_render, viewpoint_render], dim=-1)
        torchvision.utils.save_image(concat, f"{self.vcam_dbg_dir}/{viewpoint_cam.image_name}_iou{iou_score:.3f}.png")
        return iou_score