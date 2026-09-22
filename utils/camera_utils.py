#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import os, sys
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_test_dataset):
    from scene.cameras import Camera
    image = Image.open(cam_info.image_path)

    if cam_info.mask_path != "":
        try:
            alpha_mask = Image.open(cam_info.mask_path)
        except FileNotFoundError:
            print(f"Error: The mask file at path '{cam_info.mask_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.mask_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
    else:
        alpha_mask = None
       
    if cam_info.depth_path != "":
        try:
            invdepthmap = cv2.imread(cam_info.depth_path, -1)
            
            if not invdepthmap is None: 
                invdepthmap = invdepthmap.astype(np.float32) / float(2**16)
            else:
                print(f"Depth map read from {cam_info.depth_path} failed")
        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        #print(f"Depth map read from {cam_info.depth_path} failed")
        invdepthmap = None

    orig_w, orig_h = image.size

    from utils.camera_geometry import image_size
    resolution = image_size(orig_w, orig_h, args.resolution, resolution_scale)

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  primx=cam_info.primx, primy=cam_info.primy,
                  image=image, alpha_mask=alpha_mask, invdepthmap=invdepthmap, image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test, focal_length=cam_info.focal_length)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : 'Camera'):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

import torch

class CameraDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_cam_infos, args, resolution_scales, is_test):
        'Initialization'
        self.resolution_scales = resolution_scales
        self.list_cam_infos = list_cam_infos
        self.args = args
        self.args.data_device = 'cpu'
        self.is_test = is_test

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_cam_infos)

  def __getitem__(self, index):
        'Generates one sample of data'
        info = self.list_cam_infos[index]
        X = loadCam(self.args, index, info, self.resolution_scales, self.is_test)

        return X
  