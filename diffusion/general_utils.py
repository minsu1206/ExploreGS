import torch

def convert_cam_to_diff_pose(cam_set, batch=False):
    poses = []
    if batch:
        for batch in cam_set:
            poses_batch = []
            for cam in batch:
                poses_batch.append(cam.c2w)
            poses.append(torch.stack(poses_batch))
        # Nested List [[Tensor(4,4), Tensor(4,4), ...], ...]
        poses = torch.stack(poses)
    else:
        for cam in cam_set: # cam = Camera class instance
            poses.append(cam.c2w)  # append c2w
        # List [Tensor(4,4), Tensor(4,4), ...]
        poses = torch.stack(poses).unsqueeze(1)
    return poses

def convert_cam_to_diff_img(cam_set, batch=False):
    imgs = []
    if batch:
        for batch in cam_set:
            img_batch = []
            for cam in batch: # NOTE: original image = torch image tensor (C,H,W) ; refer uitls/camera_utils.py
                img_batch.append(cam.original_image)
            imgs.append(torch.stack(img_batch))
        imgs = torch.stack(imgs)
    else:
        for cam in cam_set:
            imgs.append(cam.original_image)
        imgs = torch.stack(imgs)[None]
    return imgs


def save_images_inpaint_as_grid(gt_images, masks, masked_images, pred_images, input_images, render_images, save_path):
    """
    gt_images = [B1, H, W, 3] numpy array
    masks = [B1, H, W] numpy array
    masked_images = [B1, H, W, 3] numpy array
    pred_images = [B1, H, W, 3] numpy array
    input_images = [B2, H, W, 3] numpy array
    render_images = [B1, H, W, 3] numpy array
    """
    
    img_width, img_height = pred_images[0].shape[1], pred_images[0].shape[0]
    
    ref_columns = 5
    ref_rows = (len(input_images) + ref_columns - 1) // ref_columns
    grid_img_reference = Image.new('RGB', (ref_columns * img_width, ref_rows * img_height))
    tgt_columns = 4 if gt_images is None else 5
    tgt_rows = max(len(masks), len(pred_images), len(render_images)) # all same , actually.
    grid_img_target = Image.new('RGB', (tgt_columns * img_width, tgt_rows * img_height))

    input_images = input_images.clone()
    for idx, input_image in enumerate(input_images):
        input_image += 1.0
        input_image /= 2.0
        img = Image.fromarray((input_image.cpu().to(torch.float32).numpy().transpose(1,2,0) * 255).astype(np.uint8))
        grid_x = (idx % ref_columns) * img_width
        grid_y = (idx // ref_columns) * img_height  # 2nd row
        grid_img_reference.paste(img, (grid_x, grid_y))
        
    grid_img_reference.save(save_path.replace(".png", "_ref.png"))

    for idx in range(len(render_images)):
        render_img = Image.fromarray(render_images[idx])
        mask_img = Image.fromarray(masks[idx][..., 0]).convert('RGB')
        masked_img = Image.fromarray(masked_images[idx])
        pred_img = Image.fromarray(pred_images[idx])
        gt_img = Image.fromarray(gt_images[idx]) if gt_images is not None else None
        
        grid_x_render = 0 * img_width
        grid_x_mask = 1 * img_width
        grid_x_masked = 2 * img_width
        grid_x_pred = 3 * img_width
        grid_x_gt = 4 * img_width

        grid_y = idx * img_height  # Move down by one row for each new set

        grid_img_target.paste(render_img, (grid_x_render, grid_y))
        grid_img_target.paste(mask_img, (grid_x_mask, grid_y))
        grid_img_target.paste(masked_img, (grid_x_masked, grid_y))
        grid_img_target.paste(pred_img, (grid_x_pred, grid_y))

        if gt_img is not None:
            grid_img_target.paste(gt_img, (grid_x_gt, grid_y))
    
    grid_img_target.save(save_path)
    print("[INFO] : save inpaint result as grid at ", save_path)

def crop_n_squares(imgs, crop_params=[]):
    C, H, W = imgs.shape[-3:]
    crop_size = min(H, W)
    crop_params = [] # no-use?
    if crop_params == []:
        for h in range(0, H, crop_size):
            for w in range(0, W, crop_size):
                
                if h + crop_size > H:
                    h = H - crop_size
                if w + crop_size > W:
                    w = W - crop_size
                
                crop_params.append([h, w, h+crop_size, w+crop_size])
    cropped = []
    for crop_param in crop_params:
        if len(imgs.shape) == 5: # NOT TESTED
            cropped.append(imgs[:, :, :, crop_param[0]:crop_param[2], crop_param[1]:crop_param[3]])
        # else:
        #     raise NotImplementedError("")
        elif len(imgs.shape) == 4:
            cropped.append(imgs[:, :, crop_param[0]:crop_param[2], crop_param[1]:crop_param[3]])
        elif len(imgs.shape) == 3: # NOT TESTED
            cropped.append(imgs[:, crop_param[0]:crop_param[2], crop_param[1]:crop_param[3]])
    # cropped = torch.cat(cropped)
    cropped = torch.stack(cropped, dim=1).reshape(-1, C, crop_size, crop_size)
    return cropped, crop_params
    
def center_crop(imgs):
    # imgs: training images tensor; (B,N,C,H,W) TODO:
    # center crop to make size ()
    H,W = imgs.shape[-2:]
    min_len = min(H, W)
    top = (H - min_len) // 2
    left = (W - min_len) // 2
    if len(imgs.shape) == 5:
        cropped = imgs[:, :, :, top:top+min_len, left:left+min_len]
    elif len(imgs.shape) == 4:
        cropped = imgs[:, :, top:top+min_len, left:left+min_len]
    elif len(imgs.shape) == 3:
        cropped = imgs[:, top:top+min_len, left:left+min_len]
    return cropped

def random_crop(imgs):
    B = imgs.shape[0]
    C = 3
    H, W = imgs.shape[-2:]
    crop_size = min(H, W)
    
    # Create an empty tensor for the cropped images
    if len(imgs.shape) == 3:
        B = 1
    
    cropped_imgs = torch.zeros((B, C, crop_size, crop_size), dtype=imgs.dtype, device=imgs.device)

    for i in range(B):
        top = torch.randint(0, H - crop_size + 1, (1,)).item()
        left = torch.randint(0, W - crop_size + 1, (1,)).item()
        
        cropped_imgs[i] = imgs[i, :, top:top+crop_size, left:left+crop_size]
    
    return cropped_imgs

def force_resize(imgs):
    
    C, H, W = imgs.shape[-3:]
    min_size = min(H, W)
    if len(imgs.shape) == 5:
        B, N = imgs.shape[:2]
        imgs = imgs.reshape(B*N, C, H, W)
    # F.interpolate to makes image size to be (min_size, min_size)
    resized_imgs = F.interpolate(imgs.to(dtype=torch.float32, device=imgs.device), size=(min_size, min_size), mode='bilinear', align_corners=False)
    
    return resized_imgs.to(dtype=imgs.dtype)

def gaussian_kernel(kernel_size: int, sigma: float, device):
    """Creates a 2D Gaussian kernel"""
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()  # Normalize
    return gauss

def apply_gaussian_blur(image_tensor, sigma=4):
    """Apply separable Gaussian blur to batched images using PyTorch"""
    batch_size, _, height, width = image_tensor.shape
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)  # Define kernel size
    gauss_kernel_1d = gaussian_kernel(kernel_size, sigma, image_tensor.device)
    
    # Reshape to create separable 2D kernel
    gauss_kernel_x = gauss_kernel_1d.view(1, 1, 1, -1)  # Horizontal kernel
    gauss_kernel_y = gauss_kernel_1d.view(1, 1, -1, 1)  # Vertical kernel
    
    # Apply Gaussian blur (first horizontal, then vertical) for each image in the batch
    blurred = F.conv2d(image_tensor, gauss_kernel_x, padding=(0, kernel_size // 2), groups=1)
    blurred = F.conv2d(blurred, gauss_kernel_y, padding=(kernel_size // 2, 0), groups=1)
    
    return blurred