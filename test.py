"""
All copyright reserved!
Only for reviewing

DO NOT SHARE!

 Submission 13067 CVPR
"""

from __future__ import print_function
import os
import sys
import argparse
import random
import json
import time
import datetime
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable

sys.path.append('./models/')
from model import *
from dataset import *

def torch2img(tensor,h,w):
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = tensor.reshape(3,h,w)
    img = tensor.transpose(1,2,0)
    img = img*255.
    return img



parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--gpu_ids', type=str, default='0', help='chose GPU')
opt = parser.parse_args()
print(opt)
log_name = 'IID'+'_others_'
log_path = os.path.join('./test_results/',log_name)
if not os.path.exists(log_path):
    os.makedirs(log_path)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
# load data
dataset_test = PcdIID(train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=opt.workers)
len_dataset_test = len(dataset_test)
print('len_dataset_test:', len(dataset_test))

# create network
PoIntNet = PoInt_Net(k=3) 
network = PoIntNet.cuda()
## load our pretrained model
network.load_state_dict(torch.load('./pre_trained_model/all_intrinsic.pth'))
print('start train.....')
with torch.no_grad():
    network.eval()
    start = time.time()
    for i, data in tqdm(enumerate(dataloader_test)):
        img, norms,fn = data
        img = img.cuda()
        norms = norms.cuda()
        pred_shd,pred_alb = network(img,norms,point_pos_in=1,ShaderOnly=False)
        w = 512
        h = 512
        n_b = pred_alb.shape[0]
        img_tensor = img[:,3:,:]
        img_pic = torch2img(img_tensor,h,w)
        alb_pic = torch2img(pred_alb,h,w)
        shd_pic = torch2img(pred_shd,h,w)
        img_final = np.concatenate((img_pic,alb_pic,shd_pic),axis=1) 
        cv2.imwrite(log_path+'/'+str(fn[0])+'all.png',img_final[...,::-1])
    time_use2 = time.time() - start
    print(time_use2)
