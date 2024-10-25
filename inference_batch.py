import sys
import os
import re
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = "/home/grail/camera/processed_data/10_24"
sys.path.append(BASE_DIR)

import torch
import numpy as np
from numpy import ndarray
import open3d as o3d
import time
import glob
import cv2
from numpy.linalg import inv

from models.amodal_3D_model_pointnet_plus import Amodal3DModel
from utils.stereo_custom_dataset import StereoCustomDataset
from utils.compute_box3d_iou import get_3d_box
from src.params import *

save_path = os.path.join(BASE_DIR, "results")


def downsample(pc_in_numpy: ndarray, num_object_points: int) -> ndarray:
    """downsample the pointcloud

    Parameters
    ----------
    pc_in_numpy : ndarray
        point cloud in adarray
        size [N, 6]
    num_object_points : int
        num of object points desired

    Returns
    -------
    ndarray
        downsampled pointcloud
    """
    pc_num = len(pc_in_numpy)
    idx = np.random.randint(pc_num, size=num_object_points)
    downsample_pc = pc_in_numpy[idx, :]
    return downsample_pc


def labeled_input(image_path):
    a = image_path.split("/")[-1]
    num = re.findall(r'\d+', a)
    label_dir = f'{PARENT_DIR}/datasets/labels/train/Pointcloud{num[0]}.json'
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


def point_cloud_input(pt_path_list):
    # read the pointcloud
    all_data_tensor = torch.empty(
        (0, NUM_OBJECT_POINT, 3), dtype=torch.float64)
    for pt_path in pt_path_list:
        pcd = o3d.io.read_point_cloud(pt_path)
        pc_in_numpy = np.asarray(pcd.points)
        pc_in_numpy = pc_in_numpy
        # subsample the pointcloud
        pc_in_numpy = downsample(pc_in_numpy, NUM_OBJECT_POINT)
        pc_in_tensor = torch.tensor(pc_in_numpy)
        pc_in_tensor = torch.reshape(pc_in_tensor, (1, NUM_OBJECT_POINT, 3))
        all_data_tensor = torch.cat((all_data_tensor, pc_in_tensor), 0)
    return all_data_tensor


def point_cloud_class(pt_path_list):
    pt_num = len(pt_path_list)
    one_hot = np.zeros((pt_num, NUM_SIZE_CLUSTER))
    for index, pt_path in enumerate(pt_path_list):
        categ_str = pt_path.split("/")[-1]
        categ_str = categ_str.split("_")[-1][:-4]
        categ = g_type2class[categ_str]
        one_hot[index, categ] = 1
    one_hot = torch.tensor(one_hot)
    return one_hot


def visaulization(img: ndarray, corners: list, categ, label=False):
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
    with np.load('./camera_params/camera_param_farm.npz') as X:
        mtx, Mat, tvecs = [X[i] for i in ('mtx', 'Mat', 'tvecs')]
    with np.load('./camera_params/camera_param.npz') as X:
        _, Mat, tvecs = [X[i] for i in ('mtx', 'Mat', 'tvecs')]
    if label:
        # colors = [(0, 50, 50)]
        colors = [(0, 255, 255)]
    else:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (0, 255, 255), (255, 0, 255), (125, 125, 0)]
    for index in range(len(corners)):
        corner_world = corners[index]
        corner_camera = inv(Mat) @ (corner_world.T + tvecs)
        corner_image = (mtx @ corner_camera).T
        corner = corner_image[:, :2] / corner_image[:, 2:3]
        corner = corner.astype(int)

        corner1 = corner[:4, :]
        corner2 = corner[4:8, :]
        pt1 = corner1.reshape((-1, 1, 2))
        pt2 = corner2.reshape((-1, 1, 2))
        if label:
            color = colors[index % len(colors)]
        else:
            color = colors[categ[index]]
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
        if label:
            continue
        img = cv2.addWeighted(img, alpha, zeros_mask, beta, gamma)
    return img


def main():

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Amodal3DModel()
    model.to(device)

    result_path = f"{save_path}/0819-1205/best.pt"
    result = torch.load(result_path)
    model_state_dict = result['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    image_path_list = glob.glob(
        f"{PARENT_DIR}/datasets/images/train/Image_1*")
    # image_path_list = glob.glob(
    #     f"{PARENT_DIR}/datasets/images/test/Image_*")

    t_total = 0
    for data in image_path_list:
        img_path = data
        a = data.split("/")[-1]
        num = re.findall(r'\d+', a)
        point_cloud_path = f"{PARENT_DIR}/datasets/pointclouds/train/Pointcloud{num[0]}_*"
        point_cloud_path_list = glob.glob(point_cloud_path)
        categ = point_cloud_class(point_cloud_path_list)
        features = point_cloud_input(point_cloud_path_list)
        features = features.to(device, dtype=torch.float)
        categ = categ.to(device, dtype=torch.float)

        with torch.no_grad():
            tik = time.time()
            corners = model(features, categ)
            tok = time.time()
            inference_time = (tok - tik) / len(point_cloud_path_list)
            t_total += inference_time
            print(f"inference time is {inference_time}")

        categ_numpy_onehot = categ.detach().cpu().numpy()
        categ_numpy = np.argmax(categ_numpy_onehot, axis = 1)

        img = cv2.imread(img_path)
        img = visaulization(img, corners, categ_numpy)

        label_dicts = labeled_input(img_path)
        label_corners = label2corners(label_dicts)

        img = visaulization(img, label_corners, categ_numpy, True)

        cv2.imshow("Image", img)
        k = cv2.waitKey(0)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
    print(f"average time is {t_total/len(image_path_list)}")
    print("Completed the inference!")


if __name__ == "__main__":
    main()
