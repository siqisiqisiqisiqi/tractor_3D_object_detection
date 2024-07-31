import sys
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import torch
import numpy as np
from numpy import ndarray
import open3d as o3d
import time
import glob
import cv2
from numpy.linalg import inv

from models.amodal_3D_model_transformer import Amodal3DModel
# from models.amodal_3D_model import Amodal3DModel
# from utils.stereo_custom_dataset import StereoCustomDataset
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


def main():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Amodal3DModel()
    model.to(device)

    # result_path = f"{save_path}/0714/0714_epoch45.pth"
    # result_path = f"{save_path}/0723-1816/0723-1816_epoch200.pth"

    result_path = f"{save_path}/0731-1442/last.pt"
    result = torch.load(result_path)
    model_state_dict = result['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    image_path_list = glob.glob(f"{PARENT_DIR}/datasets/images/test/Image_*")
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
            # print(f"inference time is {inference_time}")
        k = visaulization(img_path, corners)
        # print(k)
        if k == ord("q"):
            break
    print("Completed the inference!")


if __name__ == "__main__":
    main()
