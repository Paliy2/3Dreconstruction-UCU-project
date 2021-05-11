import open3d as o3d
import copy
import numpy as np
import glob

pcloud_path = "point_clouds/"
path = pcloud_path + "pcloud_%d.ply"
mesh_f_name = "meshes/my_mesh_name.obj"


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pairwise_registration(source, target):
    voxel_size = 0.05
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def main(num1=-1, num=1, matrix=None):
    # Define parameters
    voxel_size = 0.02
    threshold = 0.02

    if num1 == -1:
        image1_pcd = o3d.io.read_point_cloud("point_clouds/point_main.ply")
    else:
        image1_pcd = o3d.io.read_point_cloud(path % num1)
    image2_pcd = o3d.io.read_point_cloud(path % num)

    # todo consider this
    # # homogeneous matrix of first point cloud
    transformation_initial = np.asarray([[-0.999984, -0.00399195, 0.00400795, 1.365],
                                         [0.00399195, -0.999984, 0.00400795, -2.092],
                                         [-0.00400795, 0.00399195, 0.999984, -1.63],
                                         [0., 0., 0., 1.]])
    #
    # transformation_initial = np.asarray([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    # ])

    if matrix.all() != None:
        transformation_initial = matrix

    # Perform the Initial alignment
    print("[INFO] Initial Alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        image1_pcd, image2_pcd, threshold, transformation_initial)
    print(evaluation)

    # Normalize both point clouds using Voxel sampling to get better results
    image1_pcd.voxel_down_sample(voxel_size=voxel_size)
    image2_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Initialize the vertex information from the point clouds
    image1_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    image2_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Perform ICP Point-to-Plane registration and find true Transformation matrix
    print("[INFO] Apply point-to-plane ICP")
    registered_images = o3d.pipelines.registration.registration_icp(
        image1_pcd, image2_pcd, threshold, transformation_initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(registered_images)
    print("[INFO] Transformation Matrix:")
    print(registered_images.transformation)
    # draw_registration_result(image1_pcd, image2_pcd, registered_images.transformation)

    # Merge Point Clouds

    pcds = [image1_pcd, image2_pcd]
    print("[INFO] Full registration")

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds)

    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined = image1_pcd.transform(pose_graph.nodes[0].pose)
    pcd_combined += image2_pcd.transform(pose_graph.nodes[1].pose)
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_combined_down.remove_radius_outlier(nb_points=30, radius=1.6)
    pcd_combined_down = pcd_combined_down.select_by_index(ind)
    # cl, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    # pcd_combined_down = pcd_combined_down.select_by_index(ind)

    # o3d.visualization.draw_geometries([pcd_combined_down])
    return pcd_combined_down


def calc_rot_matrix(angle):
    c = np.cos(angle * np.pi / 180)
    c = np.cos(0 * np.pi / 180)
    s = np.sin(angle * np.pi / 180)
    s = np.sin(0 * np.pi / 180)

    return np.asarray([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])


def single_merge(files):
    pcd = o3d.io.read_point_cloud(files[0])
    for i in range(len(files) - 1):
        print(files[i])
        pcd += o3d.io.read_point_cloud(files[i])
    o3d.visualization.draw_geometries([pcd])
    return pcd


def build_mesh(pcd):
    '''
    Accept PointCloud > return Mesh
    '''
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    for alpha in np.logspace(np.log10(1.1), np.log10(0.01), num=1):
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh


def optimized_merge(files):
    pcd = o3d.io.read_point_cloud(files[0])
    for i in range(len(files) - 1):
        print(files[i])
        pcd = main(-1, i, calc_rot_matrix(i * 10))
        o3d.io.write_point_cloud("dog/point_main.ply", pcd)
    o3d.visualization.draw_geometries([pcd])
    return pcd


if __name__ == '__main__':
    files = glob.glob(pcloud_path + "*.ply")
    if len(files) < 1:
        import pcloud_generator as pg

        from_path = "images/cup/*.jpg"
        pg.process_pairwise_pclouds(from_path)
        files = glob.glob(pcloud_path + "*.ply")
    if len(files) < 1:
        raise FileNotFoundError("Please, specify correct path to point clouds e.g. <pclouds/*.pcl> ")
    for item in files:
        if "point_main" in item:
            files.remove(item)
            break

    pcd = o3d.io.read_point_cloud(path % 0)
    o3d.io.write_point_cloud("point_clouds/point_main.ply", pcd)

    # single_merge(files)
    optimized_merge(files)

    mesh = build_mesh(pcd)
    o3d.io.write_triangle_mesh(mesh_f_name, mesh)

    # pcd = o3d.io.read_point_cloud("dog/point10.ply")
    # radii = [.2, .1, .2, .4]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([rec_mesh])
