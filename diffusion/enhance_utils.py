import sys
import os
from datetime import timedelta
from tqdm import tqdm
from PIL import Image
import json
import numpy as np
import torch
import torch.nn.functional as F
import einops
from torchvision import transforms
import torchvision
from diffusion.yaml_utils import load_yaml_config
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
import gc

class EnhanceDiffusionPriorPipeline:
    def __init__(self, diff_params, args, pipe, save_path):
        self.diff_params = diff_params
        # self._debug = pipe.debug_mode  # Debugging mode
        self.prior_type = diff_params.prior_type
        self.save_path = save_path # used at debugging

        self._debug = True

        if self.prior_type == "enhancer":
            self.device = torch.device("cuda")
            model_root = diff_params.model_root
            sys.path.insert(0, model_root)
            
            from utils.utils import instantiate_from_config
            from omegaconf import OmegaConf
            from scripts.evaluation.inference import load_model_checkpoint
            from lvdm.models.samplers.ddim import DDIMSampler
            from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
            
            # from 3DGS_Enhancer_Extra/scripts/evaluation/inference.py
            diff_prior_config_path = os.path.join(model_root, diff_params.model_config_path)
            config = OmegaConf.load(diff_prior_config_path)
            model_config = config.pop("model", OmegaConf.create())
            model_config['params']['unet_config']['params']['use_checkpoint'] = False
            model = instantiate_from_config(model_config) # FIXME: import path?!
            
            model = model.to(device=self.device)
            model.perframe_ae = False # TODO: check
            resume_ckpt_path = diff_params.model_ckpt_path
            if not os.path.exists(resume_ckpt_path):
                raise ValueError("[ERROR] : checkpoint Not Found!", resume_ckpt_path)
            model = load_model_checkpoint(model, resume_ckpt_path)
            print("[INFO] : load_model_checkpoint", resume_ckpt_path)
            model.eval()
            model.requires_grad_(False)
            precision = config['lightning']['precision']
            self.weight_dtype = torch.float16 if precision == 16 else torch.float32
            print("[DEBUG] : weight_dtype = ", self.weight_dtype)
            self.width = 512 # FIXED 
            self.height = 320 # FIXED
            self.video_length = 16 # FIXED
            self.channels = model.model.diffusion_model.out_channels
            
            # bring from load_data_prompts
            transform = transforms.Compose([
                transforms.Resize((self.height, self.width)), # skip centercrop
                # transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
            
            ddim_sampler = DDIMSampler(model)
            
            self.diff_transforms = transform
            self.ddim_sampler = ddim_sampler
            self.model = model
            scene = os.path.basename(args.source_path)
            with open(os.path.join(args.source_path, f"batch_captions_result_{scene}.jsonl"), "r") as f:
                jsonl_file = json.load(f)
                self.caption = jsonl_file['response']['body']['choices'][0]['message']['content']
                print("[INFO] : caption = ", self.caption)
        else:
            raise ValueError("[ERROR]: Invalid diffusion prior type : ", self.prior_type)

        print(f"[INFO] : Load DiffusionPrior - type = {self.prior_type}")
        self.minibatch = diff_params.minibatch
        self.n_tgt_views = diff_params.n_tgt_views
        self.camera_normalization = diff_params.camera_normalization
        self.camera_relative_pose = diff_params.camera_relative_pose

    # TODO: load images from cpu to gpu per minibatch
    @torch.no_grad()
    def generate(self, pkg, save_path=""):
        """
        render_pkg : Results of GaussianRenderer ; 
            rendered img : torch.Tensor
            uncertainty map : torch.Tensor
            GT : torch.Tensor (optional ; if v_cam == test_cam)
            cam pose : torch.Tensor
        input_pkg : Reference view image & cam pose
            reference img : torch.Tensor
            cam pose : torch.Tensor
        """
        # Prepare input ; common
        # (1) dtype change
        # (2) tensor transformation
        # pkg = {"renders": renders, "cond_frame_idx": cond_frame_idx, "render_pose": render_pose, "guidance": guidance}
        renders = pkg["renders"]
        render_pose = pkg["render_pose"]
        gt_img = pkg.get("gt_img") # None or tensor
        guidances = pkg.get("guidance") # None or tensor
        # if guidances is not None:
        #     guidances = guidances.to(dtype=self.weight_dtype, device=self.device)
        cond_frame_idx = pkg.get("cond_frame_idx")
        
        H, W = renders.shape[-2:]
        prepared_img_pkg = self.prepare_img(renders, gt_img, guidances)
        
        render_pose = self.prepare_camera(render_pose, pkg.get("aux_ref_pose"))
        
        print("[CUDA] : after prepare_camera ", torch.cuda.memory_allocated()/1024**3)
        print("[CUDA] : after prepare_camera ", torch.cuda.memory_reserved()/1024**3)
        
        # import pdb; pdb.set_trace()
        
        if self.prior_type == "enhancer": # interpolation / default = minibatch mode
            renders = prepared_img_pkg.get("renders").clamp(0, 1)
            guidance = prepared_img_pkg.get("guidance") if "guidance" in prepared_img_pkg else None
            
            print("[DEBUG] : before pipeline")
            pipeline_input_pkg = {"renders":renders, 
                                  "cond_frame_idx":cond_frame_idx, # nested list # ex. [[14], [1], [5], [10], [5], [10]] 
                                  "render_pose":render_pose,
                                  "guidance":guidance}
            
            with torch.autocast("cuda"):
                gen_imgs, gen_imgs_seq_vis = self.run_pipeline_minibatch(**pipeline_input_pkg) # (B, 16, 3, 320, 512)
            if self._debug:
                pkg = {"gen":gen_imgs_seq_vis, "render":renders}
                print("[DEBUG] : save_results_v2 : ", pkg["gen"].shape, pkg["render"].shape)
                self.save_results_v2(pkg, save_path=save_path + f"/validation.png")
            # gen_imgs = torch.from_numpy(gen_imgs).reshape(-1, 3, self.height, self.width)
            gen_imgs = torch.from_numpy(gen_imgs_seq_vis).reshape(-1, 3, self.height, self.width)
        else:
            raise NotImplementedError(f"[ERROR] : DiffusionPriorPipeline - generate : {self.prior_type} type not implemented yet.")
    
        return self.post_process(gen_imgs, H=H, W=W)

    def run_pipeline(self, **kwargs):
        if self.prior_type == "enhancer":
            renders = einops.rearrange(kwargs['renders'], "(b t) c h w -> b c t h w", b=1)
            cond_frame_idx = kwargs['cond_frame_idx']
            render_pose = kwargs['render_pose']
            batch = {"renders":renders.to(dtype=self.weight_dtype, device=self.device),
                     "video":renders.to(dtype=self.weight_dtype, device=self.device),
                     "caption":[self.caption],
                     "cond_frame_idx":cond_frame_idx, # [value] ; not nested list
                     "camera_pose":render_pose[:,:3,:].reshape(-1,12)[None].to(dtype=self.weight_dtype, device=self.device), # (1, length, 12)
                     "fps":torch.tensor([5]) # fixed value
                     }
            if kwargs['guidance'] is not None:
                batch["guidance"] = kwargs['guidance']
            # print("[DEBUG] : run_pipeline : ", batch["renders"].shape, batch["guidance"].shape, batch["camera_pose"].shape, batch["cond_frame_idx"])
            log_dict = self.model.log_images(batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, unconditional_guidance_scale=1.0, mask=None, **kwargs)
            # samples = (log_dict["samples"] + 1) / 2
            samples = log_dict["samples"]
            samples = einops.rearrange(samples, "b c t h w -> b t c h w")
            samples = samples.clamp(0, 1).detach().cpu().reshape(-1, 3, 320, 512).numpy()
            samples_ = samples[[i for i in range(self.video_length) if i not in cond_frame_idx], :, :, :]
            
            return samples_, samples

    def run_pipeline_minibatch(self, **kwargs):
        if self.prior_type == "enhancer":
            batch = len(kwargs['renders'])
            gen_imgs = []
            gen_imgs_vis = []
            for i in tqdm(range(batch), desc=f"[INFO] : run_pipeline_minibatch"):
                gen_img, gen_img_vis = self.run_pipeline(renders=kwargs['renders'][i], 
                                                  cond_frame_idx=kwargs['cond_frame_idx'][i],
                                                  render_pose=kwargs['render_pose'][i],
                                                  guidance=kwargs['guidance'][i] if kwargs['guidance'] is not None else None)
                gen_imgs.append(gen_img[None])
                gen_imgs_vis.append(gen_img_vis[None])
                
                gc.collect()
                torch.cuda.empty_cache()
                
        return np.concatenate(gen_imgs, axis=0), np.concatenate(gen_imgs_vis, axis=0)

    def prepare_img(self, renders, gt_img:None, guidances:None):
        """ 
        prepare diffusion model inputs ;
        assume pred_img already has resolution as [diff_params.inpu_resolution]^2
        """
        h, w = renders.shape[-2:]
        if self.prior_type == "enhancer":
            uncertainty_batch = None
            if len(renders.shape) == 4:
                if len(renders) // self.video_length > 1:
                    batch = len(renders) // self.video_length
                    renders = renders.reshape(batch, self.video_length, 3, h, w)
                else:
                    renders = renders[None]
            renders_ = []
            for idx in range(len(renders)):
                renders_.append(self.diff_transforms_batch(renders[idx]))
            renders = torch.stack(renders_) # [-1, 1]
            output = {"renders":renders}
            # print("[DEBUG] : prepare_img : ", output["renders"].shape, cond_frame_idx)
            # prepare_img :  torch.Size([2, 16, 3, 320, 512]) 15
            # exit()
            
            if guidances is not None:
                uncertainty = []
                if len(guidances.shape) == 4:
                    if len(guidances) // self.video_length > 1:
                        batch = len(guidances) // self.video_length
                        guidances = guidances.reshape(batch, self.video_length, 1, h, w)
                    else:
                        batch = 1
                        guidances = guidances[None]
                for idx in range(len(guidances)):
                    uncertainty.append(self.diff_transforms_batch(guidances[idx]))
                uncertainty_batch = torch.stack(uncertainty)
                if len(uncertainty_batch.shape) == 4: # when batch = 1
                    uncertainty_batch = einops.rearrange(uncertainty_batch, "t (c b) h w -> b c t h w", b=1)
                elif len(uncertainty_batch.shape) == 5:
                    uncertainty_batch = einops.rearrange(uncertainty_batch, "b t c h w -> b c t h w")
                output["guidance"] = uncertainty_batch
        else:
            raise NotImplementedError(f"[ERROR] : DiffusionPriorPipeline - prepare_img : {self.prior_type} type not implemented yet.")
        
        return output

    def prepare_camera(self, render_pose, aux_ref_pose=None):
        """
        input args
            render_pose:
            aux_ref_pose: all training views' translation ; if it is not None -> use this for camera normalization
        notes
            diffusion input camera coordinate must be c2w
        """
        print("[DEBUG] : prepare_camera : ", render_pose.shape)
        if self.camera_normalization:
            if aux_ref_pose is not None:
                center, diagonal = self.get_center_and_diag(aux_ref_pose) # aux_ref_pose : (N,3)
                render_pose[..., :3, 3] -= center
                render_pose[..., :3, 3] /= diagonal

            else:
                for idx in range(len(render_pose)):
                    center, diagonal = self.get_center_and_diag(render_pose[idx, :, :3, 3])
                    render_pose[idx, :, :3, 3] -= center
                    render_pose[idx, :, :3, 3] /= diagonal

        if self.camera_relative_pose:
            render_pose = self.get_relative_pose(render_pose)

        if self.diff_params.crop == "squares": # repeat
            raise NotImplementedError("crop = squares not implemented yet")
        return render_pose

    def get_relative_pose(self, render_pose):
        # (B, N, 4, 4) or (N, 4, 4)
        
        if len(render_pose.shape) == 3:
            render_pose = render_pose[None]
        
        relative_poses = torch.zeros_like(render_pose).to(dtype=render_pose.dtype, device=render_pose.device)
        for idx, (poses) in enumerate(render_pose):
            criterion = poses[0]
            criterion_inv = torch.linalg.inv(criterion.to(dtype=torch.float32)).to(dtype=render_pose.dtype)
            relative_poses[idx] = torch.einsum('ij,kjl->kil', criterion_inv, poses)
        return relative_poses
    
    def get_center_and_diag(self, cam_centers):
        avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)
        center = avg_cam_center
        dist = torch.linalg.norm(cam_centers - center, dim=0, keepdim=True)
        diagonal = torch.max(dist)
        return center.flatten(), diagonal

    def diff_transforms_batch(self, img_batch):
        result = []
        for imgs in img_batch:
            result.append(self.diff_transforms(imgs))
        return torch.stack(result)

    def save_results_v2(self, result_pkg, save_path):
        # for enhance mode
        img_width, img_height = 512, 320
        gen_images= result_pkg["gen"].reshape(-1, 3, img_height, img_width)
        render_images = result_pkg["render"].reshape(-1, 3, img_height, img_width)

        for i, (gen_images, render_images) in enumerate(zip(result_pkg["gen"], result_pkg["render"])):
            gen_images = (gen_images.transpose(0, 2, 3, 1)*255).astype(np.uint8)
            render_images = (render_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()*255).astype(np.uint8)

            # variant of save_images_inpaint_as_grid
            ref_columns = 4
            ref_rows = (len(gen_images) + ref_columns - 1) // ref_columns
            
            grid_img_gen = Image.new('RGB', (ref_columns * img_width, ref_rows * img_height))
            grid_img_render = Image.new('RGB', (ref_columns * img_width, ref_rows * img_height))
            
            for idx, (gen_img, render_img) in enumerate(zip(gen_images, render_images)):
                grid_x = (idx % ref_columns) * img_width
                grid_y = (idx // ref_columns) * img_height
                grid_img_gen.paste(Image.fromarray(gen_img), (grid_x, grid_y))
                grid_img_render.paste(Image.fromarray(render_img), (grid_x, grid_y))
            
            grid_img_gen.save(save_path.replace(".png", f"_gen_{i}.png"))
            grid_img_render.save(save_path.replace(".png", f"_render_{i}.png"))
        
    def post_process(self, gen_imgs, H, W):
        crop = self.diff_params.crop
        gen_imgs = gen_imgs.float()
        if crop == "raw":
            return gen_imgs
        else:
            if crop == "force_resize_rot":
                gen_imgs = gen_imgs.permute(0, 1, 3, 2) # (B, C, W, H) -> (B, C, H, W)
            # Default setting
            final_output = F.interpolate(gen_imgs, size=(H, W), mode='bilinear', align_corners=False)
            return final_output