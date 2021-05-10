import open3d
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import constrained_icp as cicp

o3d = open3d

file_id_start = 0
file_id_stop = 15

voxel_size = 0.02
max_point_depth = 20
icp_dist_coarse = voxel_size * 15
icp_dist_fine = voxel_size * 5


def main():
    pcds = []
    for i in range(file_id_start, file_id_stop + 1, 1):
        pcd_file = 'dog/point%d.ply' % (i)
        # print("Reading %s..."%(pcd_file))
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcds.append(pcd)

    pcds = crop_clouds_by_depth(pcds, max_point_depth)
    # o3d.visualization.draw_geometries(pcds)

    # pcds = remove_clouds_outliers(pcds, 30, voxel_size, 1)  # removing outliers before downsample give good result.
    pcds = downsample_clouds(pcds, voxel_size)
    # pcds = remove_clouds_outliers(pcds, 5, 0.03, 1)
    estimate_clouds_normals(pcds, voxel_size * 5, 30)

    # poses = np.loadtxt('./data/poses.txt')[:, 1:]
    # transforms = translations_quaternions_to_transforms(poses)
    # pcds = transform_clouds_by_pose(pcds, transforms)

    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.5, origin = [0,0,0]) #original camera frame
    print("Showing initial cloud, pre-registration")
    # o3d.visualization.draw_geometries(pcds + [mesh_frame])
    # pose_graph = build_pose_graph(pcds, transforms, icp_dist_coarse, icp_dist_fine)
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=icp_dist_fine,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    # o3d.global_optimization(pose_graph, o3d.GlobalOptimizationLevenbergMarquardt(),
    #                         o3d.GlobalOptimizationConvergenceCriteria(), option)

    # for i in range(len(pcds)):
    #     pcds[i].transform(pose_graph.nodes[i].pose)

    o3d.visualization.draw_geometries(pcds)


def pairwise_registration(source, target, init_transform, dist_coarse, dist_fine):
    print("Apply point-to-plane ICP")
    transformation_icp, _ = cicp.cicp(source, target, init_transform, dist_coarse, dist_fine)
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, dist_fine, transformation_icp)
    return transformation_icp, information_icp


def build_pose_graph(pcds, transforms, dist_coarse, dist_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(transforms[0]))

    for i in range(1, len(pcds)):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(transforms[i]))
        # odometry = transforms[i] @ np.linalg.inv(transforms[i-1])
        transform, information = pairwise_registration(pcds[i - 1], pcds[i], np.eye(4), dist_coarse, dist_fine)
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i-1, i, transform, information, uncertain=False))

        # if i >= 2:
        #     transform, information = pairwise_registration(pcds[i - 2], pcds[i], np.eye(4), dist_coarse, dist_fine)
        #     pose_graph.edges.append(o3d.PoseGraphEdge(i - 2, i, transform, information, uncertain=True))

    transform, information = pairwise_registration(pcds[len(pcds)-1], pcds[0], np.eye(4), dist_coarse, dist_fine)
    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(len(pcds)-1, 0, transform, information, uncertain=True))
    return pose_graph


def crop_clouds_by_depth(pcds, max_depth):
    for i in range(len(pcds)):
        points = np.asarray(pcds[i].points)
        mask = points[:, 2] < max_depth
        cropped_points = points[mask, :]

        # colors = np.asarray(pcds[i].colors)
        # print(colors)
        # cropped_colors = colors[mask, :]

        pcds[i].points = o3d.utility.Vector3dVector(cropped_points)
        # pcds[i].colors = o3d.utility.Vector3dVector(cropped_colors)

    return pcds


def downsample_clouds(pcds, voxel_size):
    for i in range(len(pcds)):
        pcds[i] = o3d.geometry.PointCloud.voxel_down_sample(pcds[i], voxel_size)
    return pcds


def translations_quaternions_to_transforms(poses):
    transforms = []
    for pose in poses:
        t = pose[:3]
        q = pose[3:]

        T = np.eye(4)
        T[:3, :3] = quat2mat(q)
        T[:3, 3] = t
        transforms.append(T)
    return transforms


def transform_clouds_by_pose(pcds, transforms):
    for i in range(len(pcds)):
        pcds[i].transform(transforms[i])
    return pcds


def remove_clouds_outliers(pcds, num_points, radius, ratio=0):
    for i in range(len(pcds)):
        cl, ind = o3d.geometry.PointCloud.remove_radius_outlier(pcds[i], num_points, radius)
        if len(pcds[i].points) <= 1:
            continue
        pcd_ds = pcds[i].uniform_down_sample(0.5, pcds[i].get_min_bound(), pcds[i].get_max_bound(), False) #ind)

        if ratio > 0:
            cl, ind = o3d.geometry.PointCloud.remove_statistical_outlier(pcds[i], 50, ratio)
            pcds[i] = o3d.geometry.PointCloud.uniform_down_sample(pcds[i], ind)

    return pcds


def estimate_clouds_normals(pcds, radius, max_nn):
    for i in range(len(pcds)):
        pcds[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


if __name__ == '__main__':
    main()