import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import torch
import numpy as np
from numpy import ndarray
import open3d as o3d
import time
import glob
import random
import cv2
from numpy.linalg import inv

from models.amodal_3D_model_transformer import Amodal3DModel
# from models.amodal_3D_model import Amodal3DModel
from utils.stereo_custom_dataset import StereoCustomDataset
from torch.utils.data import DataLoader
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


def draw_text(
    img,
    text,
    uv_top_left=(5, 5),
    color=(255, 255, 255),
    fontScale=1,
    thickness=2,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def visaulization(img_dir: str, corners: list, metric: dict, idex: int):
    """Draw the 3D bounding box in the 2D image

    Parameters
    ----------
    img_dir_tuple : str
        String contains the image directory
    corners : list  
        list of coordinates of corner in world reference frame. 
        Size batchsize x 8 x 3
    metric: dict
        contains {'total loss': total_loss, 'center_loss': center_loss,
                             'heading loss': heading_loss, 'iou3d': iou3d}
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

    metric_str = ""
    for key in metric:
        value = metric[key]
        metric_str = metric_str + f"{key}: {value:1.4f}" + "\n"
    draw_text(img, metric_str)
    cv2.imshow(f"Image {idex}", img)
    k = cv2.waitKey(0)
    cv2.destroyWindow(f"Image {idex}")
    return k


def main():
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    torch.backends.cudnn.deterministic = True
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Amodal3DModel()
    model.to(device)

    result_path = f"{save_path}/0805-1713/best.pt"
    # result_path = f"{save_path}/0805-12020
    #  /best.pt"

    result = torch.load(result_path)
    model_state_dict = result['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()

    BATCH_SIZE = 1
    pc_test_path = os.path.join(PARENT_DIR, "datasets", "pointclouds", "train")
    label_test_path = os.path.join(PARENT_DIR, "datasets", "labels", "train")
    test_dataset = StereoCustomDataset(pc_test_path, label_test_path)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)
    idex = 1
    for batch, (features, test_labels, image_dir) in enumerate(test_dataloader):
        data_dicts_var = {key: value.cuda().to(torch.float)
                          for key, value in test_labels.items()}
        one_hot = data_dicts_var.get('one_hot').to(torch.float)
        features = features.to(device, dtype=torch.float)

        # for debug
        # if idex == 11:
        #     print(image_dir[0])
        #     print("test")
        # break

        with torch.no_grad():
            losses, metrics = model(features, one_hot, data_dicts_var)

        total_loss = losses['total_loss'].detach().cpu().numpy()
        center_loss = losses['center_loss'].detach().cpu().numpy()
        heading_loss = losses['heading_residual_normalized_loss'].detach(
        ).cpu().numpy()
        size_loss = losses['size_residual_normalized_loss'].detach(
        ).cpu().numpy()
        try:
            transform_loss = losses['transformer_loss'].detach().cpu().numpy()
        except:
            transform_loss = 0
        iou3d = metrics['iou3d']
        evaluation_metric = {'total loss': total_loss, 'center loss': center_loss,
                             'size loss': size_loss, 'heading loss': heading_loss,
                             'transform loss': transform_loss, 'iou3d': iou3d}
        corners = metrics['corners']

        k = visaulization(image_dir[0], corners, evaluation_metric, idex)
        if k == ord("q"):
            break
        idex = idex + 1

    print("Completed the inference!")


if __name__ == "__main__":
    main()
