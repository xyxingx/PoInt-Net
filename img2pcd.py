#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:29:45 2022

@author: Submisson 2847 ICCV
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import open3d as o3d
import time
import scipy.io
import imageio

def pltim(img,name):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(name)


def img_norm(image):
    image = image/np.max(image)
    return image.clip(0,1)


def img2pcd(img,xx,yy,z,fx,fy,cx,cy,pcd):
# pltim(img_shadow1,'dir_1')
    pcd[...,0] = (xx-cx)*z/fx
    pcd[...,1] = (yy-cy)*z/fy
    pcd[...,2] = z
    w,h,_ = pcd.shape
    # pcd_resize = cv2.resize(pcd,(256,256))
    # img_resize = cv2.resize(img,(256,256))
    pcd_resize = pcd
    img_resize = img
    # pcd_point = pcd.reshape(1024**2,3)
    # pcd_colors = img_shadow1.reshape(1024**2,3)
    pcd_point = pcd_resize.reshape(w*h,3)
    pcd_colors = img_resize.reshape(w*h,3)
    ck_point_cloud = o3d.geometry.PointCloud()
    ck_point_cloud.points = o3d.utility.Vector3dVector(pcd_point)
    ck_point_cloud.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_xyz = np.array(ck_point_cloud.points)
    pcd_rgb = np.array(ck_point_cloud.colors)
    pcd_re = np.concatenate((pcd_xyz, pcd_rgb), 1)
    return pcd_re,ck_point_cloud



def CreatDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
   
def z_grad(z):
    z_grad = z+0
    z_grad[0,...] = z_grad
   
image_name_path = '/home/xxx/Downloads/outdoor_scene/img/' 
image_name_lists = os.listdir(image_name_path)
depth_path = '/home/xxx/Downloads/outdoor_scene/depth/' 
normal_path = '/home/xxx/Datas/Outdoor-Random-3x3-new/'




save_path = '/home/xxx/Datas/NEW-Intrinsic-outdoor/pcd/pcd-ori/'
save_normal_path = '/home/xxing/Datas/NEW-Intrinsic-outdoor/gts/normal-ori/'
CreatDir(save_path)
CreatDir(save_normal_path)

all_len = len(image_name_lists)
i = 0
for fn in image_name_lists:
    if fn == '.DS_Store':
        continue
    s_time = time.time()
    fnn = fn.strip().split('.')[0]


    image_in = img_norm(np.array(cv2.imread(image_name_path+fn),'float32')[...,::-1])
    image_in = cv2.resize(image_in,(512,512))
    z = np.array(imageio.imread(depth_path+fnn+'-dpt_beit_large_512.png'),'float32')/255
    z = cv2.resize(z,(512,512))
    z = z[...,1]
    z = z+1
    image_normal = np.array(cv2.imread(normal_path+fnn+'-dpt_beit_large_512.png',-1),'float32')[...,::-1]
    image_normal = cv2.resize(image_normal,(512,512))

    
    in_z = z[z!=0]
    medians = np.median(in_z)
    
    z = z/np.max(z)
    print(fn,np.min(z),np.median(z),np.max(z))
    y = image_in.shape[0]
    x = image_in.shape[1]
    fx = x/2
    fy = y/2
    cx = x/2
    cy = y/2
    v_len = y
    u_len = x
    pcd = np.zeros((v_len,u_len,3))
    st1 = time.time()
    yy = np.arange(0,y)
    yy = np.expand_dims(yy,1).repeat(x,1)
    xx = np.arange(0,x)
    xx = np.expand_dims(xx,0).repeat(y,0)
    pcd_in,show_in = img2pcd(image_in,xx,yy,z,fx,fy,cx,cy,pcd)
    normal_in,show_normal = img2pcd(image_normal, xx, yy, z, fx, fy, cx, cy, pcd)
    ## get origanal file
    np.save(save_path+fnn, pcd_in)
    norm_np = np.array(show_normal.colors)
    np.save(save_normal_path+fnn,norm_np)
    
    e_time = time.time()-s_time
    esttime = (all_len-i)*e_time
    estm = esttime//60
    ests = esttime%60
    esth = estm//60
    estm = estm%60
    i += 1
    # o3d.visualization.draw_geometries([show_in])