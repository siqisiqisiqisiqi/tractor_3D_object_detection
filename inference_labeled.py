import sys
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import torch
import numpy as np
from numpy import ndarray
import time
import glob
import cv2
import json
from numpy.linalg import inv

from src.params import *
from utils.compute_box3d_iou import get_3d_box

save_path = os.path.join(BASE_DIR, "results")


def point_cloud_input(image_path):
    a = image_path.split("/")[-1]
    num = re.findall(r'\d+', a)
    label_dir = f'../datasets/labels/train/Pointcloud{num[0]}.json'
    with open(label_dir) as f:
        d = json.load(f)
    label_dicts = d['objects']
    return label_dicts


def label2corners(label_dicts):
    corners = []
    item_num = len(label_dicts)
    for i in range(item_num):
        label_dict = label_dicts[i]
        box_size = label_dict['dimensions']
        box_size = (box_size['length'], box_size['width'], box_size['height'])
        center = label_dict['centroid']
        center = (center['x'], center['y'], center['z'])
        heading_angle = label_dict['rotations']['z']
        corner = get_3d_box(box_size, heading_angle, center)
        corners.append(corner)
    return corners


def visaulization(img_dir: str, corners: list):
    """Draw the 3D bounding box in the 2D image

    Parameters
    ----------
    img_dir_tuple : str
        String contains the image directory
    corners : list  
        list of coordinates of corner in world reference frame. 
        Size batchsize x 8 x 3
    """
    # load the camera parameters
    with np.load('./camera_params/camera_param.npz') as X:
        mtx, Mat, tvecs = [X[i] for i in ('mtx', 'Mat', 'tvecs')]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (0, 255, 255), (255, 0, 255), (125, 125, 0)]
    img = cv2.imread(img_dir)
    for index in range(len(corners)):
        corner_world = corners[index]
        corner_camera = inv(Mat) @ (corner_world.T + tvecs)
        corner_image = (mtx @ corner_camera).T
        corner = corner_image[:, :2] / corner_image[:, 2:3]
        corner = corner.astype(int)
        # # TODO: debug why the forward is not right
        # corner[:, 0] = 1293 - corner[:, 0]

        corner1 = corner[:4, :]
        corner2 = corner[4:8, :]
        pt1 = corner1.reshape((-1, 1, 2))
        pt2 = corner2.reshape((-1, 1, 2))

        color = colors[index]
        # color = colors[0]
        thickness = 2
        cv2.polylines(img, [pt1], True, color, thickness)
        cv2.polylines(img, [pt2], True, color, thickness)
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(img, tuple(corner[i]), tuple(corner[j]), color, thickness)

        # option 2 drawing
        index1 = [1, 0, 4, 5]
        index2 = [0, 3, 7, 4]
        index3 = [2, 3, 7, 6]
        index4 = [1, 2, 6, 5]
        zero1 = np.zeros((img.shape), dtype=np.uint8)
        zero2 = np.zeros((img.shape), dtype=np.uint8)
        zero3 = np.zeros((img.shape), dtype=np.uint8)
        zero4 = np.zeros((img.shape), dtype=np.uint8)
        zero_mask1 = cv2.fillConvexPoly(zero1, corner[index1, :], color)
        zero_mask2 = cv2.fillConvexPoly(zero2, corner[index2, :], color)
        zero_mask3 = cv2.fillConvexPoly(zero3, corner[index3, :], color)
        zero_mask4 = cv2.fillConvexPoly(zero4, corner[index4, :], color)
        zeros_mask = np.array(
            (zero_mask1 + zero_mask2 + zero_mask3 + zero_mask4))

        alpha = 1
        beta = 0.55
        gamma = 0
        img = cv2.addWeighted(img, alpha, zeros_mask, beta, gamma)
    cv2.imshow("Image", img)
    k = cv2.waitKey(0)
    return k


def inference_label_visualize(num):
    # image_path_list = []
    # i = list(range(115))
    # for i in range(115):
    #     image_path = f"{PARENT_DIR}/datasets/images/Image_{i}.jpg"
    #     image_path_list.append(image_path)
    image_path_list = glob.glob(
        f"{PARENT_DIR}/datasets/images/train/Image_{num}.jpg")
    for data in image_path_list:
        img_path = data
        label_dicts = point_cloud_input(img_path)
        corners = label2corners(label_dicts)
        k = visaulization(img_path, corners)
        if k == ord("q"):
            break
    cv2.destroyAllWindows()
    print("Completed the inference!")


if __name__ == "__main__":
    num = 119
    inference_label_visualize(num)
