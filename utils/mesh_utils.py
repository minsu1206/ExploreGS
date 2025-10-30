# from DreamGaussian
import numpy as np
# import pymeshlab as pml
import open3d as o3d
import copy
import torch
from tqdm import tqdm
from collections import deque
import trimesh
from scipy.spatial import cKDTree

# Deprecated ------------------------------------------- #
# def poisson_mesh_reconstruction(points, normals=None):
#     # points/normals: [N, 3] np.ndarray

#     import open3d as o3d

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # outlier removal
#     pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

#     # normals
#     if normals is None:
#         pcd.estimate_normals()
#     else:
#         pcd.normals = o3d.utility.Vector3dVector(normals[ind])

#     # visualize
#     o3d.visualization.draw_geometries([pcd], point_show_normal=False)

#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd, depth=9
#     )
#     vertices_to_remove = densities < np.quantile(densities, 0.1)
#     mesh.remove_vertices_by_mask(vertices_to_remove)

#     # visualize
#     o3d.visualization.draw_geometries([mesh])

#     vertices = np.asarray(mesh.vertices)
#     triangles = np.asarray(mesh.triangles)

#     print(
#         f"[INFO] poisson mesh reconstruction: {points.shape} --> {vertices.shape} / {triangles.shape}"
#     )

#     return vertices, triangles


# def decimate_mesh(
#     verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
# ):
#     # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

#     _ori_vert_shape = verts.shape
#     _ori_face_shape = faces.shape

#     if backend == "pyfqmr":
#         import pyfqmr

#         solver = pyfqmr.Simplify()
#         solver.setMesh(verts, faces)
#         solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
#         verts, faces, normals = solver.getMesh()
#     else:
#         m = pml.Mesh(verts, faces)
#         ms = pml.MeshSet()
#         ms.add_mesh(m, "mesh")  # will copy!

#         # filters
#         # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
#         ms.meshing_decimation_quadric_edge_collapse(
#             targetfacenum=int(target), optimalplacement=optimalplacement
#         )

#         if remesh:
#             # ms.apply_coord_taubin_smoothing()
#             ms.meshing_isotropic_explicit_remeshing(
#                 iterations=3, targetlen=pml.Percentage(1)
#             )

#         # extract mesh
#         m = ms.current_mesh()
#         verts = m.vertex_matrix()
#         faces = m.face_matrix()

#     print(
#         f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
#     )

#     return verts, faces


# def clean_mesh(
#     verts,
#     faces,
#     v_pct=1,
#     min_f=64,
#     min_d=20,
#     repair=True,
#     remesh=True,
#     remesh_size=0.01,
# ):
#     # verts: [N, 3]
#     # faces: [N, 3]

#     _ori_vert_shape = verts.shape
#     _ori_face_shape = faces.shape

#     m = pml.Mesh(verts, faces)
#     ms = pml.MeshSet()
#     ms.add_mesh(m, "mesh")  # will copy!

#     # filters
#     ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

#     if v_pct > 0:
#         ms.meshing_merge_close_vertices(
#             threshold=pml.Percentage(v_pct)
#         )  # 1/10000 of bounding box diagonal

#     ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
#     ms.meshing_remove_null_faces()  # faces with area == 0

#     if min_d > 0:
#         ms.meshing_remove_connected_component_by_diameter(
#             mincomponentdiag=pml.Percentage(min_d)
#         )

#     if min_f > 0:
#         ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

#     if repair:
#         # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
#         ms.meshing_repair_non_manifold_edges(method=0)
#         ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

#     if remesh:
#         # ms.apply_coord_taubin_smoothing()
#         ms.meshing_isotropic_explicit_remeshing(
#             iterations=3, targetlen=pml.AbsoluteValue(remesh_size)
#         )

#     # extract mesh
#     m = ms.current_mesh()
#     verts = m.vertex_matrix()
#     faces = m.face_matrix()

#     print(
#         f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
#     )

#     return verts, faces

@torch.no_grad()
def render_set_rgbd(scene, gaussians, dataset, pipe, render_func, diffuse_only=True):
    
    if diffuse_only:
        gaussians.active_sh_degree = 0
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_cameras = scene.getTrainCameras()
    
    rgbs = []
    depths = []
    
    # Assume all training cameras are valid
    for idx, view in enumerate(tqdm(train_cameras)):
        render_pkg = render_func(view, gaussians, pipe, background)
        rgb = render_pkg["render"]
        depth = render_pkg["depth"]
        
        rgbs.append(rgb)
        depths.append(depth)
        
    return rgbs, depths

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

def tsdf_mesh(rgbs, depths, cameras, gaussians, cam_params, return_cam_info=True):
    
    depth_trunc_mode = "camera" # ["centroid-radius", "median", "camera", "adaptive"]
    
    # depth trunc v1: camera trajectory -> center -> radius
    
    if depth_trunc_mode == "camera":
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in cameras])
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        center = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        depth_trunc = radius * 2.0
        print(f"[DEBUG] : depth_trunc = {depth_trunc} radius = {radius}")
        
    elif depth_trunc_mode == "median":
        # TODO: median depth
        depths_ = []
        for i in range(len(depths)):
            depth = depths[i]
            depth_trunc = compute_depth_trunc(depth)
            depths_.append(depth_trunc)
        depth_trunc = np.median(depths_)
        
    elif depth_trunc_mode == "adaptive":
        pass
    
    elif depth_trunc_mode == "centroid-radius":
        # TODO: implement centroid-radius
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in cameras])
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        xyz = gaussians.get_xyz
        opacity = gaussians.get_opacity
        mask = (opacity >= 0.5).squeeze()
        centroid = (xyz[mask] * opacity[mask]).mean(dim=0).cpu().detach().numpy() # on world coordinate
        radius = np.linalg.norm(origins - centroid, axis=-1).min()
        depth_trunc = radius * 2.0
    
    mesh_res = cam_params.search_mesh_res # avoid OOM issue
    voxel_size = depth_trunc / mesh_res
    sdf_trunc = 5.0 * voxel_size
    num_cluster = 50
    unbounded = False
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_size,
                                                         sdf_trunc=sdf_trunc,
                                                         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    for i, cam_o3d in tqdm(enumerate(to_cam_open3d(cameras)), desc="Integrating progress"):
        rgb = rgbs[i]
        depth = depths[i]
        
        depth_trunc = depth_trunc if depth_trunc_mode in ["median", "camera"] else compute_depth_trunc(depth)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0, 1) * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
    
    mesh = volume.extract_triangle_mesh()
    cam_info = {
        "center": center,
        "radius": radius
    }
    if return_cam_info:
        return mesh, cam_info
    return mesh
    
def compute_depth_trunc(depth, max_threshold=None, min_threshold=0.5):
    depth = depth.cpu().numpy()
    valid_depths = depth[depth > 0]
    depth_trunc = np.percentile(valid_depths, 95)
    
    if max_threshold is not None:
        depth_trunc = min(depth_trunc, max_threshold)
    return max(depth_trunc, min_threshold)

def post_process_mesh(mesh, cluster_to_keep=50):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    print("post processing the mesh to have {} cluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    print(f"[DEBUG] : cluster_n_triangles = {cluster_n_triangles}")
    # print(f"[DEBUG] : cluster_area = {cluster_area}")
    cluster_area = np.asarray(cluster_area)
    cluster_to_keep = min(cluster_to_keep, len(cluster_n_triangles))
    print(f"[DEBUG] : cluster_to_keep = {cluster_to_keep}")
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def voxelize_classifiy_mesh_legacy(mesh, obb, resolution=256, surface_factor=2.0):
    """
    mesh = Open3D mesh object
    obb = open3d OrientedBoundingBox object
    """
    # mesh = Open3D mesh object
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    extents = np.asarray(obb.extent)
    rotation = np.asarray(obb.R)
    center = np.asarray(obb.center)
    
    voxel_centers_local = make_grid_from_obb(obb, resolution)
    N = len(voxel_centers_local)
    
    voxel_centers_global = (rotation @ voxel_centers_local.T).T + center
    voxel_size = (dx + dy + dz) / 3.0
    surface_distance = surface_factor * voxel_size
    
    # distances = mesh_tri.nearest.distance(voxel_centers_global)
    _, distances, _ = trimesh.proximity.closest_point(mesh_tri, voxel_centers_global)
    is_occupied = distances < surface_distance
    
    labels = np.full(N, 'unexplored', dtype=object)
    labels[is_occupied] = 'occupied'
    
    voxel_idx_3d = np.arange(N).reshape(nx, ny, nz)
    visited = np.zeros((nx, ny, nz), dtype=bool)

    def get_neighbors(i, j, k):
        nbrs = []
        for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                nbrs.append((ni, nj, nk))
        return nbrs

    # Single-ray test in +x direction => even # hits => outside, odd => inside
    def is_outside_ray_cast(point_global):
        ray_dir = np.array([[1.0, 0.0, 0.0]])
        hits = mesh_tri.ray.intersects_id(
            ray_origins=[point_global],
            ray_directions=ray_dir
        )
        return (len(hits) % 2 == 0)
    
    queue = deque()
    boundary_ijk = []
    # x-min/x-max
    boundary_ijk += [(0, j, k) for j in range(ny) for k in range(nz)]
    boundary_ijk += [(nx-1, j, k) for j in range(ny) for k in range(nz)]
    # y-min/y-max
    boundary_ijk += [(i, 0, k) for i in range(nx) for k in range(nz)]
    boundary_ijk += [(i, ny-1, k) for i in range(nx) for k in range(nz)]
    # z-min/z-max
    boundary_ijk += [(i, j, 0) for i in range(nx) for j in range(ny)]
    boundary_ijk += [(i, j, nz-1) for i in range(nx) for j in range(ny)]

    # Initialize the queue
    for (ix, iy, iz) in boundary_ijk:
        visited[ix, iy, iz] = True
        queue.append((ix, iy, iz))
        
    
    # BFS
    while queue:
        ix, iy, iz = queue.popleft()
        idx_1d = voxel_idx_3d[ix, iy, iz]

        if labels[idx_1d] != 'occupied':
            # Check outside or inside via a ray
            p_global = voxel_centers_global[idx_1d]
            if is_outside_ray_cast(p_global):
                labels[idx_1d] = 'free'
                # Explore neighbors
                for (nix, niy, niz) in get_neighbors(ix, iy, iz):
                    if not visited[nix, niy, niz]:
                        visited[nix, niy, niz] = True
                        queue.append((nix, niy, niz))
            else:
                # If "inside", do nothing => remains 'unexplored'
                pass
    
    return voxel_centers_global, labels

def voxelize_classifiy_mesh_kdtree(mesh, obb, resolution=16, surface_threshold_factor=0.5):
    """
    mesh = Open3D mesh object
    obb = open3d OrientedBoundingBox object
    """
    # mesh = Open3D mesh object
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    extents = np.asarray(obb.extent)
    rotation = np.asarray(obb.R)
    center = np.asarray(obb.center)
    
    voxel_centers_local = make_grid_from_obb(obb, resolution)
    N = len(voxel_centers_local)
    
    voxel_centers_global = (rotation @ voxel_centers_local.T).T + center
    
    verts = mesh_tri.vertices # (M, 3)
    kdtree = cKDTree(verts)
    distances, indices = kdtree.query(voxel_centers_global, workers=-1) #(N^3 resolution voxel)
    
    diagonal_length = np.linalg.norm(extents)
    surface_threshold = (diagonal_length / resolution) * surface_threshold_factor

    occupied_mask = distances < surface_threshold
    inside_mask = mesh_tri.contains(voxel_centers_global)
    occupied_mask = occupied_mask | inside_mask
    occupied = voxel_centers_global[occupied_mask]
    unknown_mask = ~occupied_mask & ~inside_mask
    unknown = voxel_centers_global[unknown_mask]
    free_mask = ~ (occupied_mask | unknown_mask)
    free = voxel_centers_global[free_mask]
    
    print(f"[DEBUG] : free {len(free)} unknown {len(unknown)} occupied {len(occupied)}")
    
    labels = np.full(N, 'unknown', dtype=object)
    labels[occupied_mask] = 'occupied'
    labels[free_mask] = 'free'
    
    return voxel_centers_global, labels

def export_voxel_grid_as_ply(voxel_centers, labels, ply_filename="voxel_labels.ply", use_frontier=False):
    """
    Export voxel centers (N, 3) with color-coded labels as a PLY point cloud.
    - free       -> green  (0,1,0)
    - occupied   -> red    (1,0,0)
    - unexplored -> blue   (0,0,1)
    """
    # 1. Create an Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    
    # Assign positions
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    
    # 2. Create color array
    colors = np.zeros((len(labels), 3))  # Initialize color array
    colors[labels == 'free'] = [0, 1, 0]       # Green for free
    colors[labels == 'occupied'] = [1, 0, 0]   # Red for occupied
    colors[labels == 'unknown'] = [0, 0, 1] # Blue for unexplored

    # Gradient from red to yellow // use for debugging
    # for i in range(len(labels)):
    #     ratio = i / (len(labels) - 1)  # Normalize index to [0, 1]
    #     colors[i] = np.array([ratio, ratio, ratio])  # Interpolate between red and yellow

    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 3. Export to PLY
    o3d.io.write_point_cloud(ply_filename, pcd)
    
    # free only
    pcd_free = pcd.select_by_index(np.where(labels == 'free')[0].tolist())
    free_ply_filename = ply_filename.replace(".ply", "_free.ply")
    o3d.io.write_point_cloud(free_ply_filename, pcd_free)
    
    if use_frontier > 0:
        pcd_frontier = pcd.select_by_index(np.where(labels == 'frontier')[0].tolist())
        pcd_frontier.colors = o3d.utility.Vector3dVector(np.repeat(np.array([[1, 0, 1]]), len(pcd_frontier.points), axis=0)) # magenta
        
        frontier_ply_filename = ply_filename.replace(".ply", "_frontier.ply")
        o3d.io.write_point_cloud(frontier_ply_filename, pcd_frontier)
    
def export_mesh_w_value_as_ply(voxel_centers, values, ply_filename="mesh_w_value.ply"):
    """
    Export mesh with color-coded values (either scale or visibility) as a PLY mesh.
    - Color represents values: Blue (low) → Red (high)
    
    Arguments:
        voxel_centers (np.array): (N, 3) voxel positions.
        values (np.array): (N,) values to encode as color (e.g., occupied scale or visibility).
        ply_filename (str): Output filename for PLY file.
    """

    # Normalize values to [0, 1]
    values_norm = (values - values.min()) / (values.max() - values.min())

    # Create color mapping: Blue (low) → Red (high)
    colors = np.zeros((len(voxel_centers), 3))  # Initialize color array
    colors[:, 0] = values_norm  # Red channel
    colors[:, 2] = 1 - values_norm  # Blue channel (inverse of red)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save as PLY
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"Occupied space visualization saved to {ply_filename}")

def make_grid_from_obb(obb, resolution):
    center = np.asarray(obb.center)
    extents = np.asarray(obb.extent)
    rotation = np.asarray(obb.R)
    half_extents = extents / 2.0
    
    nx = ny = nz = resolution
    dx = extents[0] / nx
    dy = extents[1] / ny
    dz = extents[2] / nz
    
    local_xs = np.linspace(-half_extents[0], half_extents[0], nx, endpoint=False) + 0.5 * dx
    local_ys = np.linspace(-half_extents[1], half_extents[1], ny, endpoint=False) + 0.5 * dy
    local_zs = np.linspace(-half_extents[2], half_extents[2], nz, endpoint=False) + 0.5 * dz
    
    X, Y, Z = np.meshgrid(local_xs, local_ys, local_zs, indexing="ij")
    voxel_centers_local = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return voxel_centers_local