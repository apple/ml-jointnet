#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import numpy as np
import cv2
import torch

def read_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)[..., None].astype('float32') / 255

def read_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1].astype('float32') / 255

def read_depth_map(path):
    arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if arr.dtype == np.uint16:
        arr = arr.astype('float32') / 65535
    elif arr.dtype == np.uint8:
        arr = arr.astype('float32') / 255
    else:
        raise ValueError
    return arr

def get_dummy_mask(ref_arr, fill_value):
    return np.full_like(ref_arr[:,:,:1], fill_value)

def to_tensor(arr, rescale=False):
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).clamp(0,1)
    if rescale:
        t = t * 2 - 1
    return t
