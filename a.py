import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import structure
import processor
import features

def dino(im_1, im_2):
    # Dino
    img1 = cv2.imread(im_1)
    img2 = cv2.imread(im_2)
    pts1, pts2 = features.find_correspondence_points(img1, img2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)

    #fig, ax = plt.subplots(1, 2)
    #ax[0].autoscale_view('tight')
    #ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    #ax[0].plot(points1[0], points1[1], 'r.')
    #ax[1].autoscale_view('tight')
    #ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    #ax[1].plot(points2[0], points2[1], 'r.')
    #fig.show()

    height, width, ch = img1.shape
    intrinsic = np.array([
        [2071, 0, width / 2],
        [0, 2205, height / 2],
        [0, 0, 1]])


    return points1, points2, intrinsic


points3d = np.empty((0,0))
files = glob.glob("dog_images/*.png")
files = files[:21]
length = len(files)

for item in range(length-1):
    points1, points2, intrinsic = dino(files[item], files[(item+1)%length])

    points1n = np.dot(np.linalg.inv(intrinsic), points1)
    points2n = np.dot(np.linalg.inv(intrinsic), points2)
    E = structure.compute_essential_normalized(points1n, points2n)
    print('Computed essential matrix:', (-E / E[0][1]))

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
    tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
    # tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)

    if not points3d.size:
        points3d = tripoints3d
    else:
        points3d = np.concatenate((points3d, tripoints3d), 1)


fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.gca(projection='3d')

ax.plot(points3d[0], points3d[1], points3d[2], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()

import  pyvista as pv


points3d = points3d[:3]

cloud = pv.PolyData(points3d.reshape(len(points3d[0]), 3))
cloud.plot()

volume = cloud.delaunay_3d(alpha=2.)
shell = volume.extract_geometry()
shell.plot()
