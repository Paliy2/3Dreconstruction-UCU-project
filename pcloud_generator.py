import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import features, structure, processor
import open3d as o3d
import glob
import os
from utils import camera


SHOW_PLOTS = True

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
    input("Press to continue: ")


def get_features(img1, img2, intrinsic):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img1 = cv2.filter2D(img1, -1, kernel)
    img2 = cv2.filter2D(img2, -1, kernel)

    pts1, pts2 = features.find_correspondence_points(img1, img2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)
    if SHOW_PLOTS:
        plot_two_pointsets(img1, img2, points1, points2)
        input("Press to continiue: ")

    height, width, ch = img1.shape
    # between 0.7 and width

    arr = [[1.37580713e+03, 0.00000000e+00, 3.68183020e+02],
     [0.00000000e+00, 1.37058017e+03, 5.17882529e+02],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    # arr = [[1.04251417e+03, 0.00000000e+00, 3.31015284e+02],
    #        [0.00000000e+00, 1.04954921e+03, 6.57064833e+02],
    #        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]


    # intrinsic = np.array(arr)
    return points1, points2, intrinsic


def get_p1_p2_matrices(points1n, points2n, E):
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
    return P1, P2


def plot3D(tripoints3d):
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()


def get_point_cloud(img1, img2, matrix):
    points1, points2, intrinsic = get_features(img1, img2, matrix)

    # normalize points
    points1n = np.dot(np.linalg.inv(intrinsic), points1)
    points2n = np.dot(np.linalg.inv(intrinsic), points2)
    # essential matrix
    E = structure.compute_essential_normalized(points1n, points2n)

    # At camera P1, calculate the parameters for camera P2
    # Getting 4 possible camera parameters
    P1, P2 = get_p1_p2_matrices(points1n, points2n, E)

    tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)
    # plot3D(tripoints3d)

    tripoints_trasposed = np.transpose(tripoints3d[:3])
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(tripoints_trasposed)
    return pcl

def process_pairwise_pclouds(from_path, matrix):
    clear_cache("point_clouds/")

    files = glob.glob(from_path)
    if files == []:
        print("Please, specify correct path for images. <e.g. 'imgs/*.png'>")

    for i in range(len(files) - 1):
        print("Processing ", files[i])
        pcl1 = get_point_cloud(files[i], files[i + 1], marix)
        if SHOW_PLOTS:
            o3d.visualization.draw_geometries([pcl1])
        # input("Press to continiue: ")
        o3d.io.write_point_cloud("point_clouds/pcloud_%d.ply" % i, pcl1)
    return "Success!"


def clear_cache(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def calibrate(renew_imgs=False):
    chess_board_images = "images/board/*.jpg"
    if renew_imgs:
        # todo test this row (actualy it will never be used :)
        clear_cache("/".join(chess_board_images.split("/")[:-1]))

    camera.calibrate(chess_board_images)


if __name__ == '__main__':
    print("You need to make a photos of chessboard to calibrate your camera. ")
    print("Load images to the ./images/board directory.")
    loaded_board_images = False

    # todo TRUE param will delete current chessboard images
    matrix = calibrate(renew_imgs=loaded_board_images)
    marix = [[980.33821028, 0., 349.98499292],
           [0., 978.53235453, 658.69],
           [0., 0., 1.]]

    global_path = "images/*.jpg"
    # ("Dog_RGB/*.JPG")
    process_pairwise_pclouds(global_path, matrix)
