#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:35:40 2022

gen_normal2

@author: Submisson 2847 ICCV
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, cv2
import scipy.io
import pandas as pd
import imageio
import matplotlib.pyplot as plt
# plt.set_cmap('jet')
TAG_FLOAT = 202021.25
def get_points_coordinate(depth, instrinsic_inv, device="cuda"):
    B, height, width, C = depth.size()
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
    depth_xyz = xyz * depth.reshape(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth


def pltim(img,name):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(name)

def read_pfm(path):
    """Read pfm file.
    Args:
        path (str): path to file
    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N
if __name__ == '__main__':
    ## step.1 input
    # depth & intrinsic path
    depth_path = '/home/xxx/Downloads/img/depth/' # your depth path
    file_lists = os.listdir(depth_path)
    depth_list = file_lists
    normal_map_path = '/home/xxx/Datas/Outdoor-Random-3x3-new/' # your normal save path
    if not os.path.isdir(normal_map_path):
        os.makedirs(normal_map_path)
    if not os.path.isdir(normal_map_npy_path):
        os.makedirs(normal_map_npy_path)
    # load depth & intrinsic
    for file_name in depth_list:
        fnn = file_name.strip().split('.')[0]
        if file_name == '.DS_Store':
            continue
        if not file_name.endswith('.pfm'):
            continue
        # fnn = 'alley_2_frame_0020'
        # file_name = '9-dpt_beit_large_512.pfm'
        # load depth & intrinsic
        depth, scale = read_pfm(depth_path + file_name)
        depth = depth / np.max(depth)
        ## depth can also load from png file (filter needed)
        # z = np.array(imageio.imread(depth_path+fnn+'.png'),'float32')/255
        # get_sobel = cv2.Sobel(z,cv2.CV_32F,1,0,ksize=3)
        # pltim(get_sobel,'depth')
        # z = z[...,1]
        # # z = cv2.GaussianBlur(z,(15,15),0)
        #
        # z = cv2.blur(z,(12,12))
        # # z = 1-z
        # depth_np = 1-z
        depth_np = depth
        # depth_np = np.round(depth_np,)
        where_nan = np.isnan(depth_np)
        depth_np[where_nan]= 0
        
        in_z = depth_np[depth_np!=0]
        pltim(depth_np, "depth")
        H, W = depth_np.shape
        fx = H/2
        fy = W/2
        cx = H/2
        cy = W/2
        cam_ins = np.array([fx,0,cx,
                            0,fy,cy,
                            0,0,1
            ],'float32').reshape((3,3))
        depth_torch = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(-1) # (B, h, w, 1)
        valid_depth = depth_np > 0.0
        intrinsic_np = np.array(cam_ins,'float32')
        intrinsic_inv_np = np.linalg.inv(intrinsic_np)
        intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0) # (B, 4, 4)
    
        ## step.2 compute matrix A
        # compute 3D points xyz
        # depth_torch = depth_torch
        print(depth_torch.shape)
        points = get_points_coordinate(depth_torch, intrinsic_inv_torch[:, :3, :3], "cpu")
        point_matrix = F.unfold(points, kernel_size=3, stride=1, padding=1, dilation=1)
    
        # An = b
        matrix_a = point_matrix.view(1, 3, 9, H, W)  # (B, 3, 25, HxW)
        matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
        matrix_a_trans = matrix_a.transpose(3, 4)
        matrix_b = torch.ones([1, H, W, 9, 1])
    
        # dot(A.T, A)
        point_multi = torch.matmul(matrix_a_trans, matrix_a)
        matrix_deter = torch.det(point_multi.to("cpu"))
        # make inversible
        inverse_condition = torch.ge(matrix_deter, 1e-18)
        inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
        inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
        # diag matrix to update uninverse
        diag_constant = torch.ones([3], dtype=torch.float32)
        diag_element = torch.diag(diag_constant)
        diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        diag_matrix = diag_element.repeat(1, H, W, 1, 1)
        # inversible matrix
        inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
        inv_matrix = torch.inverse(inversible_matrix.to("cpu"))
    
        ## step.3 compute normal vector use least square
        # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
        generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
        norm_normalize = F.normalize(generated_norm, p=2, dim=3)
        norm_normalize_np = norm_normalize.squeeze().cpu().numpy()
    
        ## step.4 save normal vector
        # np.save(depth_path.replace("depth", "normal"), norm_normalize_np)
        norm_normalize_draw = (((norm_normalize_np + 1) / 2))
        # np.save(normal_map_npy_path_ori+fn,norm_normalize_draw)
        # norm_normalize_draw_64 = cv2.resize(norm_normalize_draw,(64,64))
        # np.save(normal_map_npy_path+fnn,norm_normalize_draw)
        
        # pltim(norm_normalize_draw, 'normal')
        cv2.imwrite(normal_map_path+fnn+'.png', norm_normalize_draw[...,::-1]*255.)
