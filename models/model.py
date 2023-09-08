from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from math import *
import torch.nn.functional as F

import sys

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 36)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        # print(x.shape)
        x = x.view(-1, 1024)
        # print(x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1]).astype(np.float32))).view(1, 36).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 6, 6)
        return x



class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        # print('trans.shape',trans.shape)
        x = x.transpose(2, 1)
        # print('x.shape',x.shape)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet_IID(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNet_IID, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = F.relu(x)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat



class PointNet_IID_2(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNet_IID_2, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = torch.tanh(x)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


class PointNet_IID_3(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNet_IID_3, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = torch.tanh(x)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

class PointNet_IID_shd_2(nn.Module):
    """
    input: point cloud
    out: predict light direction per point (first)
    calculate shading (second)
    final render shading (thrid)
    para: ShaderOnly: check shader
    
    """
    def __init__(self, k = 2, feature_transform=False):
        super(PointNet_IID_shd_2, self).__init__()
        self.pos_pred = PointNet_IID_2(k,feature_transform=False)
        self.shader = PointNet_IID_3(k,feature_transform=False)
    
    def forward(self,x,img_normal,point_pos_in=1,ShaderOnly=False):
        if not ShaderOnly:
            point_pos,_,_ = self.pos_pred(x)
        else:
            point_pos = point_pos_in
        point_posn_norm = torch.nn.functional.normalize(point_pos,dim=1)
        shd_cal = point_posn_norm[:,0,:]*img_normal[:,0,:]+point_posn_norm[:,1,:]*img_normal[:,1,:]+point_posn_norm[:,2,:]*img_normal[:,2,:]
        # # print(shd_cal.shape)
        shd_cal = shd_cal.unsqueeze(1)
        shd_cal = shd_cal.repeat(1,3,1)
        shd_cal_in = torch.cat((point_posn_norm,img_normal),1)
        # print(shd_cal_in.shape)
        final_shd,_,_ = self.shader(shd_cal_in)
        return point_posn_norm,shd_cal,final_shd

class PoInt_Net(nn.Module):
    """
    input: point cloud
    out: predict light direction per point 
        final render shading 
        Albedo  AlbedoNet
    para: ShaderOnly: check shader
    Note: the paras are pre-train and loaded

    
    """
    def __init__(self, k = 2,feature_transform=False):
        super(PoInt_Net, self).__init__()
        self.shadingnet = PointNet_IID_shd_2(k,feature_transform=False)
        self.albedoNet = PointNet_IID(k,feature_transform=False)

    
    def forward(self,x,img_normal,point_pos_in=1,ShaderOnly=False):
        alb_pred,_,_ = self.albedoNet(x)
        _,_,final_shd = self.shadingnet(x,img_normal,point_pos_in=1,ShaderOnly=False)
        return final_shd,alb_pred