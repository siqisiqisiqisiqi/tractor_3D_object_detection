import sys
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import open3d as o3d
import numpy as np
import glob
import json
from numpy.linalg import inv
from open3d_line_mesh import LineMesh

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


def point_cloud_input(path):
    a = path.split("/")[-1]
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


def visual3d(pc_path, corners):
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
              (0, 1, 1), (1, 0, 1), (0.5, 0.5, 0)]
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1, origin=[0, 0, 0])
    mesh = o3d.io.read_point_cloud(pc_path)

    boxes = []
    vis_objs = [mesh]
    for i, corner in enumerate(corners):
        color = colors[i % len(colors)]
        boxes.append(bbox_obj_mesh(corner, color=color))

    # o3d.visualization.draw_geometries([mesh, boxes[0][0]])
    vis_core(vis_objs, boxes)


# def bbox_obj(points, color=[1, 0, 0]):
#     colors = [color for i in range(len(LINES))]
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(points),
#         lines=o3d.utility.Vector2iVector(LINES),
#     )
#     line_set.colors = o3d.utility.Vector3dVector(colors)
#     return line_set

def bbox_obj_mesh(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_mesh1 = LineMesh(points, LINES, colors, radius=0.02)
    line_mesh1_geoms = line_mesh1.cylinder_segments
    return line_mesh1_geoms


def vis_core(plys, boxes):

    vis = o3d.visualization.Visualizer()
    vis.create_window()

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


def main():
    pc_path_list = glob.glob(
        f"{PARENT_DIR}/datasets/original_point_cloud/train/Pointcloud203.ply")
    for data in pc_path_list:
        pc_path = data
        label_dicts = point_cloud_input(pc_path)
        corners = label2corners(label_dicts)
        k = visual3d(pc_path, corners)
    print("Completed the inference!")


if __name__ == "__main__":
    main()
