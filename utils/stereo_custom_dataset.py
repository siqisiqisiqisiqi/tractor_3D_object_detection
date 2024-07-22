import sys
import os
import re
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

import torch
import json
import numpy as np
import open3d as o3d
from typing import Tuple
from numpy import ndarray
from torch.utils.data import Dataset
from numpy.linalg import norm
from torch.utils.data import DataLoader

from src.params import *
from utils.model_util import angle_thres

pc_train_path = os.path.join(PARENT_DIR, "datasets", "pointclouds", "train")
label_train_path = os.path.join(PARENT_DIR, "datasets", "labels", "train")
pc_test_path = os.path.join(PARENT_DIR, "datasets", "pointclouds", "test")
label_test_path = os.path.join(PARENT_DIR, "datasets", "labels", "test")


class StereoCustomDataset(Dataset):
    def __init__(self, pc_path: str, label_path: str, downsample=True):
        """custom dataset

        Parameters
        ----------
        pc_path : str
            input point cloud path 
        label_path : str
            labeled data path
        downsample : bool, optional
            downsample pointcloud flag, by default True
        """
        super().__init__()

        self.pc_path = pc_path
        self.label_path = label_path
        self.DS = downsample

        self.pc_list = glob(f"{pc_path}/*.ply")

    def downsample(self, pc_in_numpy: ndarray, num_object_points: int) -> ndarray:
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

    def __len__(self):
        return len(self.pc_list)

    def convertlabelformat(self, label: dict, label_dir: str) -> Tuple[dict, str]:
        """convert the labeled 3D bounding box in the desired format

        Parameters
        ----------
        label : dict
            label data from the file
        label_dir : str
            label data directory

        Returns
        -------
        Tuple[dict, str]
            label2: label data in the desired format
            img_dir: img directory correspond to the pointcloud
        """
        center = label['centroid']
        box3d_center = np.array([center['x'], center['y'], center['z']])
        size_class = np.array([g_type2onehotclass[label['name']]])
        standard_size = g_type_mean_size[label['name']]
        size = label['dimensions']
        box_size = np.array(
            [size['length'], size['width'], size['height']])
        size_residual = standard_size - box_size
        angle = label['rotations']['z']
        angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
        angle = angle_thres(angle)
        angle_class = np.array([angle // angle_per_class])
        angle_residual = np.array([angle % angle_per_class])
        object_class = g_type2onehotclass[label['name']]
        one_hot = np.zeros(NUM_SIZE_CLUSTER)
        one_hot[object_class] = 1
        x = label_dir.split("/")
        x[-2] = "images"
        a = re.findall(r'\d+', x[-1])
        num = a[0]
        x[-1] = f"Image_{num}.jpg"
        img_dir = "/".join(x)
        label2 = {'one_hot': one_hot, 'box3d_center': box3d_center,
                  'size_class': size_class, 'size_residual': size_residual,
                  'angle_class': angle_class, 'angle_residual': angle_residual}
        return label2, img_dir

    def __getitem__(self, index: int) -> Tuple[ndarray, dict, str]:
        """getitem function for the custom dataset

        Parameters
        ----------
        index : _type_
            input data index

        Returns
        -------
        Tuple[ndarray, dict, str]
            pc_in_numpy: downsampled point cloud in ndarray
            label2: deisred label data
            img_dir: corresponding image directory 
        """
        pcd = o3d.io.read_point_cloud(self.pc_list[index])
        pc_in_numpy = np.asarray(pcd.points)
        centroid_point = np.sum(pc_in_numpy, 0) / len(pc_in_numpy)
        pc_name = self.pc_list[index].split("/")[-1].split("_")
        label_dir = f"{self.label_path}/{pc_name[0]}.json"
        pc_class = pc_name[4].split(".")[0]
        with open(label_dir) as f:
            d = json.load(f)

        object_num = len(d['objects'])
        distance = []
        min_dist = 1e4
        min_idx = 0
        for i in range(object_num):
            center = d['objects'][i]['centroid']
            label_center = np.array([center['x'], center['y'], center['z']])
            label_class = d['objects'][i]['name']
            if label_class == pc_class:
                distance = norm(label_center - centroid_point, 2)
                if distance < min_dist:
                    min_dist = distance
                    min_idx = i
        if min_dist == 1e4:
            print("Can't find labels!!!!!!!")
        label = d['objects'][min_idx]
        pc_in_numpy = pc_in_numpy
        if self.DS:
            pc_in_numpy = self.downsample(pc_in_numpy, NUM_OBJECT_POINT)
        label2, img_dir = self.convertlabelformat(label, label_dir)
        return pc_in_numpy, label2, img_dir


if __name__ == "__main__":
    BATCH_SIZE = 8
    train_dataset = StereoCustomDataset(pc_train_path, label_train_path)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    test_dataset = StereoCustomDataset(pc_test_path, label_test_path)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    for batch, (train_features, train_labels, _) in enumerate(train_dataloader):
        # print(train_features.shape)
        pass
        # break
