from __future__ import print_function
import os
import cv2
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.utils.data as data




class PcdIID(data.Dataset):
    def __init__(self, train=False, foldn=0,sizes=16):
        self.file_list = os.listdir('./Data/pcd/pcd-ori/')
        self.train = train

    def __getitem__(self, index):
        fn = self.file_list[index]
        fnn = fn.strip().split('.')[0]
        pcd = np.load('./Data/pcd/pcd-ori/'+fnn+'.npy')
        norms = np.load('./Data/gts/normal-ori/'+ fnn + '.npy')
            
        p,c_p = pcd.shape
    
        pcd = np.array(pcd, dtype='float32')
        

        norms = np.array(norms,'float32')
        pcd = pcd.transpose(1,0)
        norms = norms.transpose(1,0)
        norms = torch.from_numpy(norms.copy())
        pcd = torch.from_numpy(pcd.copy())
        return pcd,norms,fnn


    def __len__(self):
        return (len(self.file_list))


if __name__ == '__main__':
    dataset = PcdIID(train=True)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=6)
    for ep in range(10):
        time1 = time.time()
        for i, data in enumerate(dataload):
            img, normal, fn = data
            print(img.shape)
            print(normal.shape)
