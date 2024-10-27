import sys
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
# PARENT_DIR = "/home/grail/camera/processed_data/10_24"
sys.path.append(BASE_DIR)

import open3d as o3d
import numpy as np
import glob
import json
import torch
from numpy.linalg import inv
from open3d_line_mesh import LineMesh
from models.amodal_3D_model_pointnet_plus import Amodal3DModel

from src.params import *
from utils.compute_box3d_iou import get_3d_box

save_path = os.path.join(BASE_DIR, "results")
LINES = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
]

def downsample(pc_in_numpy, num_object_points):
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

def point_cloud_input(path):
    a = path.split("/")[-1]
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


def visual3d(pc_path, corners_labeled, corners, categ):
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
    #           (0, 1, 1), (1, 0, 1), (0.5, 0.5, 0)]
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0),
            (1, 1, 0), (1, 0, 1), (0, 0.5, 0.5)]
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1, origin=[0, 0, 0])
    mesh = o3d.io.read_point_cloud(pc_path)

    boxes = []
    vis_objs = [mesh]
    for i, corner in enumerate(corners):
        # color = colors[i % len(colors)]
        color = colors[categ[i]]
        boxes.append(bbox_obj_mesh(corner, color=color))

    for i, corner in enumerate(corners_labeled):
        color = (1,1,0)
        boxes.append(bbox_obj_mesh(corner, color=color))
    # o3d.visualization.draw_geometries([mesh, boxes[0][0]])
    vis_core(vis_objs, boxes)


def bbox_obj_mesh(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_mesh1 = LineMesh(points, LINES, colors, radius=0.02)
    line_mesh1_geoms = line_mesh1.cylinder_segments
    return line_mesh1_geoms


def vis_core(plys, boxes):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Visualization', width=1920, height=1080)

    for ply in plys:
        vis.add_geometry(ply)
    for box in boxes:
        for line in box:
            vis.add_geometry(line)
    render_option = vis.get_render_option()
    render_option.line_width = 3.0
    render_option.point_size = 2.0

    vis.run()
    vis.destroy_window()

def init_model(result_path):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Amodal3DModel()
    model.to(device)

    result = torch.load(result_path)
    model_state_dict = result['model_state_dict']
    model.load_state_dict(model_state_dict)
    return model, device

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

def batch_point_cloud_input(pt_path_list):
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

def main():
    result_path = f"{save_path}/0819-1205/best.pt"
    model, device = init_model(result_path)
    model.eval()
    
    pc_path_list = glob.glob(
        f"{PARENT_DIR}/datasets/original_point_cloud/train/Pointcloud203.ply")
    for data in pc_path_list:
        pc_path = data
        label_dicts = point_cloud_input(pc_path)
        labeled_corners = label2corners(label_dicts)

        a = data.split("/")[-1]
        num = re.findall(r'\d+', a)[0]
        point_cloud_path = f"{PARENT_DIR}/datasets/pointclouds/train/Pointcloud{num}_*"

        point_cloud_path_list = glob.glob(point_cloud_path)
        categ = point_cloud_class(point_cloud_path_list)
        features = batch_point_cloud_input(point_cloud_path_list)
        features = features.to(device, dtype=torch.float)
        categ = categ.to(device, dtype=torch.float)

        with torch.no_grad():
            corners = model(features, categ)

        categ_numpy_onehot = categ.detach().cpu().numpy()
        categ_numpy = np.argmax(categ_numpy_onehot, axis = 1)

        visual3d(pc_path, labeled_corners, corners, categ_numpy)
    print("Completed the inference!")


if __name__ == "__main__":
    main()
