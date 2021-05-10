import matplotlib.pyplot as plt
import numpy as np
import cv2
import structure
import processor
import features
import open3d as o3d
import glob


def plot_points_only(p1, p2):
    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].plot(p1[0], p1[1], 'r.')
    ax[1].plot(p2[0], p2[1], 'r.')
    plt.show()


def plot_two_pointsets(img1, img2, points1, points2):
    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].plot(points1[0], points1[1], 'r.')
    ax[1].autoscale_view('tight')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].plot(points2[0], points2[1], 'r.')
    fig.show()


def get_features(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img1 = cv2.filter2D(img1, -1, kernel)
    img2 = cv2.filter2D(img2, -1, kernel)

    pts1, pts2 = features.find_correspondence_points(img1, img2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)
    plot_two_pointsets(img1, img2, points1, points2)
    input()
    height, width, ch = img1.shape
    # between 0.7 and width
    arr = np.array([
        [1000, 0, width / 2],
        [0, 700, height / 2],
        [0, 0, 1]])

    # arr = [[1.04251417e+03, 0.00000000e+00, 3.31015284e+02],
    #        [0.00000000e+00, 1.04954921e+03, 6.57064833e+02],
    #        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    arr = [[980.33821028, 0., 349.98499292],
           [0., 978.53235453, 658.69],
           [0., 0., 1.]]
    intrinsic = np.array(arr)
    return points1, points2, intrinsic


if __name__ == '__main__':
    def get_point_cloud(img1, img2):
        points1, points2, intrinsic = get_features(img1, img2)

        # normalize points
        points1n = np.dot(np.linalg.inv(intrinsic), points1)
        points2n = np.dot(np.linalg.inv(intrinsic), points2)
        E = structure.compute_essential_normalized(points1n, points2n)
        # print('Computed essential matrix:', (-E / E[0][1]))

        # At camera 1, calculate the parameters for camera 2
        # Get 4 possible camera parameters
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
        tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)

        # plot_points_only(points1n, points2n)

        # fig = plt.figure()
        # fig.suptitle('3D reconstructed', fontsize=16)
        # ax = fig.gca(projection='3d')
        # ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
        # ax.set_xlabel('x axis')
        # ax.set_ylabel('y axis')
        # ax.set_zlabel('z axis')
        # ax.view_init(elev=135, azim=90)
        # plt.show()

        tripoints_trasposed = tripoints3d[:3]
        tripoints_trasposed = np.transpose(tripoints_trasposed)

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(tripoints_trasposed)
        # pcl.points = o3d.utility.Vector3dVector(tripoints3d.reshape(len(tripoints3d[0]), 3))
        return pcl


    # files = glob.glob("Dog_RGB/*.JPG")
    files = glob.glob("images/*.jpg")

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

    for i in range(len(files) - 1):
        print(files[i])
        pcl1 = get_point_cloud(files[i], files[i + 1])
        # input()
        # o3d.visualization.draw_geometries([pcl1])
        # input("Press to continiue: ")

        o3d.io.write_point_cloud("dog/point%d.ply" % i, pcl1)
