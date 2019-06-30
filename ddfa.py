#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np

import torch
import torch.utils.data as data
import cv2
import pickle
import argparse
from .io import _numpy_to_tensor, _load_cpu, _load_gpu
from .params import *


#from skimage.transform import rotate
from skimage.feature import local_binary_pattern
#from skimage import data, io
#from skimage.color import label2rgb
#import skimage
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
import cv2
from skimage.morphology import disk
import skimage.filters.rank as sfr
import scipy.signal as signal
# settings for LBP
import torch
radius = 3
n_points = 4 * radius




def _parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp


def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean

    p, offset, alpha_shp, alpha_exp = _parse_param(param)

    if dense:
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]
    else:
        """For 68 pts"""
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex

def img_loader(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image

def img_loader2(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image =sfr.bottomhat(image, disk(2))  #半径为2的圆形滤波器
    R = image
    G = image
    B = image
    image = cv2.merge([B, G, R])


    return image


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class DDFADataset(data.Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.img_loader = img_loader
        self.img_loader2 =img_loader2

    def _target_loader(self, index):
        target = self.params[index]

        return target

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)
        img2 = self.img_loader2(path)

        #cv2.imshow('img',img)
        #print('-------------------------')
        #cv2.waitKey (0)  
        
        #cv2.destroyAllWindows() 
        

        target = self._target_loader(index)

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        return img, img2, target

    def __len__(self):
        return len(self.lines)


class DDFATestDataset(data.Dataset):
    def __init__(self, filelists, root='', transform=None):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = img_loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.lines)
