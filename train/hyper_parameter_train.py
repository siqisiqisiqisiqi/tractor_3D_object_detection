import sys
import os
from typing import Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

import time
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from models.amodal_3D_model_pointnet_plus import Amodal3DModel
# from models.amodal_3D_model_transformer import Amodal3DModel
# from models.amodal_3D_model import Amodal3DModel
import itertools
from utils.stereo_custom_dataset import StereoCustomDataset
from src.params import *
from src.pytorchtools import EarlyStopping
from src.f import combine_dicts_to_list
import json

pc_train_path = os.path.join(PARENT_DIR, "datasets", "pointclouds", "train")
label_train_path = os.path.join(PARENT_DIR, "datasets", "labels", "train")

pc_test_path = os.path.join(PARENT_DIR, "datasets", "pointclouds", "train")
label_test_path = os.path.join(PARENT_DIR, "datasets", "labels", "train")

save_path = os.path.join(ROOT_DIR, "results")

# select the device
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test(model: Amodal3DModel, loader: DataLoader) -> Tuple[dict, dict]:
    """_summary_

    Parameters
    ----------
    model : Amodal3DModel
        3D object detection model
    loader : DataLoader
        test_dataloader

    Returns
    -------
    Tuple[dict, dict]
        return the test loss and iou metric
    """
    test_losses = {
        'total_loss': 0.0,
        'center_loss': 0.0,
        'heading_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'iou_value_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0,
        'mask_center_loss': 0.0,
        'transformer_loss': 0.0,
    }

    test_metrics = {
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }

    model.eval()
    for i, (features, label_dicts, _) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):

        data_dicts_var = {key: value.cuda().to(torch.float)
                          for key, value in label_dicts.items()}
        one_hot = data_dicts_var.get('one_hot').to(torch.float)
        features = features.to(device, dtype=torch.float)

        with torch.no_grad():
            losses, metrics = model(features, one_hot, data_dicts_var)

        for key in test_losses.keys():
            if key in losses.keys():
                test_losses[key] += losses[key].detach().item()
        for key in test_metrics.keys():
            if key in metrics.keys():
                test_metrics[key] += metrics[key]

    for key in test_losses.keys():
        test_losses[key] /= len(loader)
        test_losses[key] = round(test_losses[key], 5)
    for key in test_metrics.keys():
        test_metrics[key] /= len(loader)
        test_metrics[key] = round(test_metrics[key], 5)

    return test_losses, test_metrics


def plot_result(iou3d_value: list, path: str):

    plt.plot(iou3d_value, c='b')
    # plt.plot(test_loss, c='r')
    plt.legend(['iou3d set'])
    plt.title('iou3d')
    plt.ylabel('iou3d')
    plt.xlabel('# of epoch')
    plt.grid()
    plt.savefig(f"{path}/result.jpg")
    plt.show()


def train():
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_dataset = StereoCustomDataset(pc_train_path, label_train_path)
    test_dataset = StereoCustomDataset(pc_test_path, label_test_path)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)

    strtime = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    strtime = strtime[4:13]

    result_path = f"{save_path}/{strtime}_grid_search"
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)
    # center_w = [0.5, 0.8, 1, 1.2, 2.0]
    # size_w = [0.5, 0.8, 1, 1.2, 2.0]
    # heading_w = [0.5, 0.8, 1, 1.2, 2.0]
    iou_heading_w = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
    # hyper_parameter_list = list(itertools.product(
    # center_w, size_w, heading_w))
    hyper_parameter_list = list(itertools.product(iou_heading_w))
    iou3d_list = []
    index = 0
    for hyper_parameter in hyper_parameter_list:

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        model = Amodal3DModel(hyper_parameter=hyper_parameter)
        model.to(device)

        # define the optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=BASE_LR,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEPS, gamma=GAMMA)

        early_stopping = EarlyStopping(
            patience=50, verbose=True, path=f"{result_path}/early_stopping.pt")

        best_iou3d = 0.0
        train_total_losses_data = []
        test_total_losses_data = []
        train_save_dic = {}
        test_save_dic = {}
        for epoch in range(MAX_EPOCH):
            train_losses = {
                'total_loss': 0.0,
                'center_loss': 0.0,
                'heading_class_loss': 0.0,
                'heading_residual_normalized_loss': 0.0,
                'iou_value_loss': 0.0,
                'size_residual_normalized_loss': 0.0,
                'stage1_center_loss': 0.0,
                'corners_loss': 0.0,
                'mask_center_loss': 0.0,
                'transformer_loss': 0.0,
            }
            train_metrics = {
                'iou2d': 0.0,
                'iou3d': 0.0,
                'iou3d_0.7': 0.0,
            }
            model.train()
            for i, (features, label_dicts, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9):

                data_dicts_var = {key: value.cuda().to(torch.float)
                                  for key, value in label_dicts.items()}

                features = features.to(device, dtype=torch.float)
                one_hot = data_dicts_var.get('one_hot').to(torch.float)
                losses, metrics = model(features, one_hot, data_dicts_var)
                total_loss = losses['total_loss']
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                for key in train_losses.keys():
                    if key in losses.keys():
                        train_losses[key] += losses[key].detach().item()
                for key in train_metrics.keys():
                    if key in metrics.keys():
                        train_metrics[key] += metrics[key]

            for key in train_losses.keys():
                train_losses[key] /= len(train_dataloader)
                train_losses[key] = round(train_losses[key], 5)
            for key in train_metrics.keys():
                train_metrics[key] /= len(train_dataloader)
                train_metrics[key] = round(train_metrics[key], 5)
            train_epoch_dic = combine_dicts_to_list(
                train_losses, train_metrics)
            train_save_dic = combine_dicts_to_list(
                train_save_dic, train_epoch_dic)
            print(
                f"Finished the {epoch} epoch train +++++++++++++++++++ Total train loss is {train_losses['total_loss']}")
            train_total_losses_data.append(train_losses['total_loss'])

            test_losses, test_metrics = test(model, test_dataloader)
            test_epoch_dic = combine_dicts_to_list(test_losses, test_metrics)
            test_save_dic = combine_dicts_to_list(
                test_save_dic, test_epoch_dic)
            print(
                f"Finished the {epoch} epoch test +++++++++++++++++++ Total test loss is {test_losses['total_loss']}")
            test_total_losses_data.append(test_losses['total_loss'])
            scheduler.step()

            print(f"learning rate is {scheduler.get_lr()[0]}")
            if scheduler.get_lr()[0] < MIN_LR:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = MIN_LR

            if test_metrics['iou3d'] >= best_iou3d:
                best_iou3d = test_metrics['iou3d']
                best_state = {
                    'epoch': epoch + 1,
                    'test_iou3d': test_metrics['iou3d'],
                    'test_iou2d': test_metrics['iou2d'],
                    'hyper_parameter': hyper_parameter,
                }
            # early_stopping needs the validation loss to check if it has decresed
            early_stopping(test_losses['total_loss'], model)
            if epoch < 200:
                early_stopping.early_stop = False
            if early_stopping.early_stop and epoch > 200:
                print("Early stopping")
                break

        iou3d_list.append(best_iou3d)
        index = index + 1
        savepath = f"{result_path}/index_{index}_hyper_parameter_{hyper_parameter}.json"
        with open(savepath, 'w') as json_file:
            json.dump(best_state, json_file)

    plot_result(iou3d_list, result_path)


if __name__ == "__main__":

    train()
