import matplotlib.pyplot as plt
import numpy as np
import cv2
import structure
import processor
import features
# import pyvista as pv
import open3d as o3d


# cam = open3d.camera.PinholeCameraIntrinsic()
# cam.intrinsic_matrix = [[2360, 0.00, 1] , [0.00, 2360, 1], [0.00, 0.00, 1.00]]

# pcd = open3d.geometry.PointCloud.create_from_depth_image(
#     'imgs/dinos/7.jpg', cam)
# cv2.imshow(pcd)

def dino(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    pts1, pts2 = features.find_correspondence_points(img1, img2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)

    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].plot(points1[0], points1[1], 'r.')
    ax[1].autoscale_view('tight')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].plot(points2[0], points2[1], 'r.')
    fig.show()

    height, width, ch = img1.shape
    # between 0.7 and width
    intrinsic = np.array([
        [2071, 0, width / 2],
        [0, 2205, height / 2],
        [0, 0, 1]])

    return points1, points2, intrinsic


if __name__ == '__main__':
    def get_point_cloud(img1, img2):
        points1, points2, intrinsic = dino(img1, img2)

        # Calculate essential matrix with 2d points.
        # Result will be up to a scale

        # First, normalize points
        points1n = np.dot(np.linalg.inv(intrinsic), points1)
        points2n = np.dot(np.linalg.inv(intrinsic), points2)
        E = structure.compute_essential_normalized(points1n, points2n)
        # print('Computed essential matrix:', (-E / E[0][1]))

        # Given we are at camera 1, calculate the parameters for camera 2
        # Using the essential matrix returns 4 possible camera paramters
        P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        P2s = structure.compute_P_from_essential(E)

        ind = -1
        for i, P2 in enumerate(P2s):
            # Find the correct camera parameters
            d1 = structure.reconstruct_one_point(
                points1n[:, 0], points2n[:, 0], P1, P2)

            # Convert P2 from camera view to world view
            P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
            d2 = np.dot(P2_homogenous[:3, :4], d1)

            if d1[2] > 0 and d2[2] > 0:
                ind = i

        P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
        # tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
        tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)

        pcl = o3d.geometry.PointCloud()

        # pcl.points = tripoints3d
        tripoints3d = tripoints3d[:3]
        pcl.points = o3d.utility.Vector3dVector(tripoints3d.reshape(len(tripoints3d[0]), 3))
        return pcl


    # path_1 = "dog_images/im_0001.png"
    # path_2 = "dog_images/im_0002.png"

    # path_3 = "dog_images/im_0003.png"
    import glob

    files = glob.glob("Dog_RGB/*.JPG")
    # files = ["imgs/dinos/10.jpg", "imgs/dinos/11.jpg", "imgs/dinos/12.jpg",
    #          "imgs/dinos/13.jpg", "imgs/dinos/14.jpg", ]

    # img = o3d.io.read_image("dog_images/im_0001.png")
    #
    # cam = o3d.camera.PinholeCameraIntrinsic()
    # cam.intrinsic_matrix = np.array([
    #     [2071, 0, 1280 / 2],
    #     [0, 2205, 1024 / 2],
    #     [0, 0, 1]])
    #
    # cam = o3d.camera.PinholeCameraIntrinsic(
    #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    #
    # pcl = o3d.geometry.PointCloud.create_from_rgbd_image(img)
    #
    # o3d.visualization.draw_geometries(pcl)

    for i in range(len(files)-1):
        print(files[i])
        pcl1 = get_point_cloud(files[i], files[i+1])
        input("Press to continiue: ")
        o3d.io.write_point_cloud("dog/point%d.ply" % i, pcl1)


    # for i in range(1, len(files) - 1):
    #     pcl2 = get_point_cloud(files[i], files[i + 1])
    #     # pcl1 = o3d.visualization.draw_geometries([pcl1, pcl2])
    #     pcl1 += pcl2
    #
    # o3d.io.write_point_cloud("point.ply", pcl1)
