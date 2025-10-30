import numpy as np
import torch
import math
import open3d as o3d
import torch.nn.functional as F

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def make_sphere_view_directions(azi_angle, elev_angle, cam_pos, mean_up_vec=None):
    """
    Make view directions on a sphere
    cam_pos : camera position
    azi_angle : azimuth angle (degree)
    elev_angle : elevation angle (degree)
    """
    up_vec = np.array([0.0, 1.0, 0.0])
    refine_R = None
    if mean_up_vec is not None:
        refine_R = rotation_matrix_from_vecs(up_vec, mean_up_vec)

    azi_rad = np.deg2rad(azi_angle)
    elev_rad = np.deg2rad(elev_angle)
    
    azi_rad_range = np.linspace(0, 2 * np.pi, int(2 * np.pi / azi_rad))
    elev_rad_range = np.linspace(0, np.pi, int(np.pi / elev_rad))
    
    views = []
    # TODO: omit the case when elev = 90 or 180 for efficiency
    for azi in azi_rad_range:
        for elev in elev_rad_range:
            
            x = np.cos(elev) * np.sin(azi)
            y = np.sin(elev)
            z = np.cos(elev) * np.cos(azi)
            
            forward = np.array([x, y, z])
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(up_vec, forward)
            right = right / np.linalg.norm(right)
            
            up = np.cross(forward, right)
            up = up / np.linalg.norm(up)
            
            R = np.column_stack((right, up, forward))
            T = cam_pos
            
            # Check for inversion
            camera_up = R[:, 1]
            if np.dot(camera_up, up_vec) < 0:
                # Option 1: Flip the up vector
                R[:, 1] *= -1
                R[:, 0] = np.cross(R[:, 1], R[:, 2])  # Recompute right vector
                R[:, 2] = np.cross(R[:, 0], R[:, 1])  # Recompute forward vector
            
            c2w = np.eye(4)
            c2w[:3, :3] = R if refine_R is None else refine_R @ R
            c2w[:3, 3] = T

            views.append(c2w)
    return views

# def modify_pose_upvec(c2w, up_vec):
#     # modify the up vector of the pose
#     forward = c2w[:3, 2]
#     forward = forward / np.linalg.norm(forward)
    
#     right = np.cross(up_vec, forward)
#     right = right / np.linalg.norm(right)
    
#     up = np.cross(forward, right)
#     up = up / np.linalg.norm(up)
    
#     c2w[:3, :3] = np.column_stack((right, up, forward))
#     return c2w


def modify_pose_rightvec(c2w, right_vec):
    # modify the up & right vectors of the pose
    forward = c2w[:3, 2]
    forward = forward / np.linalg.norm(forward)
    
    right = right_vec
    right = right / np.linalg.norm(right)
    
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    
    c2w[:3, :3] = np.column_stack((right, up, forward))
    return c2w

def modify_pose_upvec_sign(c2w, up_vec):
    # if c2w upvec is opposite to up_vec, flip the upvec
    if np.dot(c2w[:3, 1], up_vec) < 0:
        c2w[:3, 1] *= -1
    return c2w

def make_look_center_view_directions(cam_pos, center, angle):
    """
    Make view directions focusing towards a center point within a given angle constraint.

    Args:
        cam_pos (np.array): Camera position (shape: (3,))
        center (np.array): Target center position (shape: (3,))
        angle (float): Maximum deviation angle from the direct viewing direction (in degrees)

    Returns:
        List of 4x4 camera-to-world transformation matrices.
    """
    up_vec = np.array([0.0, 1.0, 0.0])  # Default up vector
    view_dir = center - cam_pos  # Compute direction from cam_pos to center
    view_dir = view_dir / np.linalg.norm(view_dir)  # Normalize

    # Define hemisphere
    right_vec = np.cross(up_vec, view_dir)
    if np.linalg.norm(right_vec) < 1e-6:  # Handle edge case where up_vec is parallel to view_dir
        right_vec = np.array([1.0, 0.0, 0.0])
    right_vec /= np.linalg.norm(right_vec)
    up_vec = np.cross(view_dir, right_vec)
    up_vec /= np.linalg.norm(up_vec)

    start_c2w = np.eye(4)
    start_c2w[:3, :3] = np.column_stack((right_vec, up_vec, view_dir))
    start_c2w[:3, 3] = cam_pos

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Sample azimuth and elevation angles within the defined cone
    num_samples_azi = int(2 * np.pi / angle_rad)
    num_samples_elev = int(np.pi / angle_rad)

    azi_range = np.linspace(-angle_rad, angle_rad, num_samples_azi)
    elev_range = np.linspace(-angle_rad, angle_rad, num_samples_elev)

    views = []

    for azi in azi_range:
        for elev in elev_range:
            # Compute small perturbations around the view direction
            rot_azi = np.array([
                [np.cos(azi), -np.sin(azi), 0],
                [np.sin(azi), np.cos(azi), 0],
                [0, 0, 1]
            ])

            rot_elev = np.array([
                [1, 0, 0],
                [0, np.cos(elev), -np.sin(elev)],
                [0, np.sin(elev), np.cos(elev)]
            ])

            perturbed_dir = rot_azi @ rot_elev @ view_dir
            perturbed_dir /= np.linalg.norm(perturbed_dir)

            # Recompute right and up vectors
            right = np.cross(up_vec, perturbed_dir)
            right /= np.linalg.norm(right)
            up = np.cross(perturbed_dir, right)
            up /= np.linalg.norm(up)

            # Construct transformation matrix
            R = np.column_stack((right, up, perturbed_dir))
            T = cam_pos

            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T

            views.append(c2w)
    print("[DEBUG] : make_look_center_view_directions : ", len(views))
    return views, start_c2w

def make_look_center_view_directions_refine(cam_pos, center, angle, train_cameras):
    """
    Generate view directions focusing towards a center point within a given angle constraint,
    but refine the up/right vectors based on the nearest training camera.

    Args:
        cam_pos (np.array): Camera position (shape: (3,))
        center (np.array): Target center position (shape: (3,))
        angle (float): Maximum deviation angle from the direct viewing direction (in degrees)
        train_cameras (List): List of cameras, each with a .c2w 4x4 matrix (torch or numpy).

    Returns:
        List of 4x4 camera-to-world transformation matrices.
    """
    # -------------------------------------------------------------------------
    # 1) Find the nearest training camera, and use its right & up vectors
    # -------------------------------------------------------------------------
    c2ws = np.array([cam.c2w.numpy() for cam in train_cameras])  # (N, 4, 4)
    train_positions = c2ws[:, :3, 3]                              # (N, 3)
    
    # Get index of nearest camera to 'cam_pos'
    closest_idx = np.linalg.norm(train_positions - cam_pos, axis=-1).argmin()
    closest_cam = train_cameras[closest_idx]

    # Extract the right (x-axis) and up (y-axis) vectors from the nearest camera
    # By convention here:
    #   - c2w[:3, 0] is the 'right' vector
    #   - c2w[:3, 1] is the 'up' vector
    #   - c2w[:3, 2] is the 'forward' (view) vector
    closest_rightvec = closest_cam.c2w[:3, 0].numpy()
    closest_upvec    = closest_cam.c2w[:3, 1].numpy()

    # -------------------------------------------------------------------------
    # 2) Compute the primary viewing direction from cam_pos to center
    # -------------------------------------------------------------------------
    view_dir = center - cam_pos
    view_dir /= np.linalg.norm(view_dir)

    # We'll start by using the nearest camera's right & up vectors as a basis.
    # However, we need to ensure these vectors are orthonormal w.r.t. the new view_dir.
    right_vec = closest_rightvec
    right_vec /= np.linalg.norm(right_vec)

    # Re-derive the up vector so that it is consistent with view_dir and right_vec.
    up_vec = np.cross(view_dir, right_vec)
    up_vec /= np.linalg.norm(up_vec)

    start_c2w = np.eye(4)
    start_c2w[:3, :3] = np.column_stack((right_vec, up_vec, view_dir))
    start_c2w[:3, 3] = cam_pos
    # -------------------------------------------------------------------------
    # 3) Set up angle sampling for azimuth/elevation around the main view_dir
    # -------------------------------------------------------------------------
    angle_rad = np.deg2rad(angle)

    # You can adjust these sample densities to your liking
    # or replicate what you did in make_look_center_view_directions
    num_samples_azi = int(2 * np.pi / angle_rad)
    num_samples_elev = int(np.pi / angle_rad)

    azi_range = np.linspace(-angle_rad, angle_rad, num_samples_azi)
    elev_range = np.linspace(-angle_rad, angle_rad, num_samples_elev)

    # -------------------------------------------------------------------------
    # 4) For each (azi, elev) pair, perturb the main view_dir, then reconstruct
    #    right/up vectors so each final camera transform is well-formed
    # -------------------------------------------------------------------------
    views = []
    for azi in azi_range:
        for elev in elev_range:
            # Rotation around the (approx) up vector (azi) 
            rot_azi = np.array([
                [np.cos(azi), -np.sin(azi), 0],
                [np.sin(azi),  np.cos(azi), 0],
                [0,           0,            1]
            ])

            # Rotation around the (approx) right vector (elev)
            rot_elev = np.array([
                [1,           0,            0],
                [0,  np.cos(elev), -np.sin(elev)],
                [0,  np.sin(elev),  np.cos(elev)]
            ])

            # Compose these small rotations
            # Note: rot_azi & rot_elev are in a coordinate frame where z=forward
            # If you want them in world-space, you might need to transform them, 
            # but for small angles it might be fine.
            perturbed_dir = rot_azi @ rot_elev @ view_dir
            perturbed_dir /= np.linalg.norm(perturbed_dir)

            # Recompute right & up
            # We keep the original "closest_upvec" in spirit but re-calc 
            # so that everything remains orthonormal around the new perturbed_dir
            right = np.cross(up_vec, perturbed_dir)
            if np.linalg.norm(right) < 1e-9:
                # Fallback if cross is degenerate
                right = closest_rightvec
            right /= np.linalg.norm(right)

            up = np.cross(perturbed_dir, right)
            up /= np.linalg.norm(up)

            # Construct c2w
            R = np.column_stack((right, up, perturbed_dir))
            T = cam_pos
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3]  = T

            views.append(c2w)

    print(f"[DEBUG] : make_look_center_view_directions_refine : {len(views)} views created.")
    return views, start_c2w

def create_free_space_mesh(free_space_voxels, cam_positions, voxel_size=0.01):
    # Create voxel cubes for each free space voxel
    voxel_meshes = []
    for voxel in free_space_voxels:
        # Create a cube for each voxel
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        cube.translate(voxel - np.array([voxel_size/2, voxel_size/2, voxel_size/2]))  # Center the cube
        cube.paint_uniform_color([0, 1, 0])  # Color: green for free space
        voxel_meshes.append(cube)

    # Create small spheres for camera positions
    camera_meshes = []
    for cam_pos in cam_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * 2) # small margin
        sphere.paint_uniform_color([1, 0, 0])  # Color: red for camera positions
        sphere.translate(cam_pos)
        camera_meshes.append(sphere)

    # Combine all meshes
    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in voxel_meshes + camera_meshes:
        combined_mesh += mesh

    return combined_mesh

def line_in_free_space(ptA, ptB, free_voxels, labels, step=0.01):
    """ Checks if the line from ptA -> ptB intersects any 'occupied' or 'unknown' voxels.
    ptA (np.array): shape (3,)
    ptB (np.array): shape (3,)
    free_voxels (np.ndarray): Nx3 array of voxel centers labeled as free, occupied, unknown.
    labels (np.ndarray): same length as free_voxels. 
    step (float): sampling step along the line in world units.
    """
    # In practice, you'd do a more efficient check using spatial data structures (k-d tree, octree).
    # For simplicity, we do a naive line sampling.
    direction = ptB - ptA
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return True
    
    direction_unit = direction / length
    n_steps = int(np.ceil(length / step))
    
    for k in range(n_steps+1):
        t = k / n_steps
        sample_point = ptA + t * direction
        # Check if sample_point is in free space
        if not in_free_space(sample_point, free_voxels, labels):
            return False
    
    return True

def compute_viewing_direction(cam_intrinsics, cam_extrinsics, pixel_coords, img_size):
    """
    Computes the viewing direction for given pixel coordinates.

    Args:
        cam_intrinsics (torch.Tensor): 3x3 camera intrinsic matrix.
        cam_extrinsics (torch.Tensor): 4x4 camera extrinsic matrix (world to camera).
        pixel_coords (list of tuples): [(y1, x1), (y2, x2)] list of pixel positions.
        img_size (tuple): (H, W) original image size.

    Returns:
        torch.Tensor: (2, 3) viewing direction in world space.
    """
    H, W = img_size
    K_inv = torch.inverse(cam_intrinsics)  # Inverse intrinsics

    # Convert pixel coordinates to normalized device coordinates
    pixel_coords = torch.tensor(pixel_coords, dtype=torch.float32)
    yx_norm = torch.stack([
        (pixel_coords[:, 1] - cam_intrinsics[0, 2]) / cam_intrinsics[0, 0],  # (x - cx) / fx
        (pixel_coords[:, 0] - cam_intrinsics[1, 2]) / cam_intrinsics[1, 1],  # (y - cy) / fy
        torch.ones(pixel_coords.shape[0])  # Z = 1 for ray direction
    ], dim=-1)  # (N, 3)
    
    # Convert to world space if extrinsics are given
    if cam_extrinsics is not None:
        R = cam_extrinsics[:3, :3]  # Extract rotation
        d_world = (R @ yx_norm.T).T  # Transform ray direction
    else:
        d_world = yx_norm  # Keep in camera space if no extrinsics

    return F.normalize(d_world, dim=-1)  # Normalize to unit vectors

def rot_c2w_right(c2w, angle):
    angle = np.radians(angle)
    R = np.array([[np.cos(angle), 0, -np.sin(angle), 0],
                  [0, 1, 0, 0],
                  [np.sin(angle), 0, np.cos(angle), 0],
                  [0, 0, 0, 1]])
    return c2w @ R

def rot_c2w_up(c2w, angle):
    angle = np.radians(angle)
    R = np.array([[1, 0, 0, 0],
                  [0, np.cos(angle), np.sin(angle), 0],
                  [0, -np.sin(angle), np.cos(angle), 0],
                  [0, 0, 0, 1]])
    return c2w @ R

def axis_angle_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0],
                  [y*x*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0],
                  [z*x*(1-c)-y*s, z*y*(1-c)+x*s, z*z*(1-c)+c, 0],
                  [0, 0, 0, 1]])
    return R

def rotation_matrix_from_vecs(source_vec, target_vec):
    """
    Returns a 3x3 rotation matrix that rotates 'source_vec' to 'target_vec'.
    Both vectors should be normalized.
    """
    source_vec = source_vec / np.linalg.norm(source_vec)
    target_vec = target_vec / np.linalg.norm(target_vec)

    # If vectors are nearly identical, no rotation is needed
    if np.allclose(source_vec, target_vec, atol=1e-7):
        return np.eye(3)

    # Axis of rotation
    axis = np.cross(source_vec, target_vec)
    axis = axis / np.linalg.norm(axis)
    # Angle between vectors
    angle = np.arccos(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))

    # Rodrigues' rotation formula
    K = np.array([
        [0,         -axis[2],  axis[1]],
        [axis[2],   0,        -axis[0]],
        [-axis[1],  axis[0],   0      ]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R