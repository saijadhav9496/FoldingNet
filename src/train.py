#!/usr/bin/env python
'''
train foldingnet

author  : Ruoyu Wang; Yuqiong Li
created : 10/25/18 1:29 PM
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim

from foldingnet import FoldingNetVanilla

from foldingnet import FoldingNetShapes
from foldingnet import ChamferDistance
from foldingnet import ChamfersDistance3
from datasets import pcdDataset
from torch.utils.data import DataLoader
import numpy as np
from utils import check_exist_or_remove
from utils import check_exist_or_mkdirs
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
print("HAHAHA")

'''
def train(dataset, model, batch_size, lr, epoches, log_interval, save_along_training):
    """train implicit version of foldingnet
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    chamfer_distance_loss = ChamfersDistance3()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model = model.train().cuda()   # set attributes
    #check_exist_or_remove("../log/train_loss_log.txt")
    check_exist_or_mkdirs("../log/train_loss_log.txt")
    loss_log = open('../log/train_loss.txt', 'w')
    for ep in range(0, epoches):
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            opt.zero_grad()
            data = batch.cuda()
            points_pred = model(data)
            loss = chamfer_distance_loss(data, points_pred)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if batch_idx % log_interval == log_interval - 1:
                print('[%d, %5d] loss: %.6f' %
                    (ep + 1, batch_idx + 1, running_loss / log_interval))
                print('[%d, %5d] loss: %.6f' %
                    (ep + 1, batch_idx + 1, running_loss / log_interval), file=loss_log)
                running_loss = 0.0
        if save_along_training:
            torch.save(model.state_dict(), os.path.join('../model', 'ep_%d.pth' % ep))
    if save_along_training:   # the last one
        torch.save(model.state_dict(), os.path.join('../model', 'ep_%d.pth' % ep))
    loss_log.close()
    return


if __name__ == '__main__':
    print("In main")
    train = 0
    evaluate = 1
    # ROOT = "../data/nyc/"    # root path
    #TRIAN_PATH = "../data/catelog/train.txt"
    
    ROOT = "/home/skulgod/sai/FoldingNet/data/point_clouds"    # root path
    TRIAN_PATH = "/home/skulgod/sai/FoldingNet/data/modelnet40_train.txt"
    MLP_DIMS = (3,64,64,64,128,1024)
    FC_DIMS = (1024, 512, 512)
    FOLDING1_DIMS = (521, 512, 512, 3)   # change the input feature of the first fc because now has 9 dims instead of 2
    FOLDING2_DIMS = (515, 512, 512, 3)
    MLP_DOLASTRELU = False
    
    if not os.path.exists('../model'):
            os.makedirs('../model')
    kwargs = {
        'lr': 0.0001,
        'epoches': 330,
        'batch_size': 16,
        'log_interval': 10,
        'save_along_training': True
    }

    with open(TRIAN_PATH) as fp:
        catelog = fp.readlines()
    catelog = [x.strip() for x in catelog]
    print("catelog done !")

    dataset = pcdDataset(ROOT, catelog)
    model = FoldingNetShapes(MLP_DIMS, FC_DIMS, FOLDING1_DIMS, FOLDING2_DIMS)

    
    if(train):
        print("Training")
        train(dataset, model, **kwargs)
        print("End training!!!")
        
    elif(evaluate):
        print("Evaluating")
        model.load_state_dict(torch.load("/home/skulgod/sai/FoldingNet/model/ep_329.pth"))
        model.eval().cuda()
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
        chamfer_distance_loss = ChamfersDistance3()
        for batch_idx, batch in enumerate(dataloader):
            if(batch_idx ==1):
                pcd_in1 = batch[1,:,:]
                pcd_in2 = batch[3,:,:]
                pcd_in3 = batch[7,:,:]
                data = batch.cuda()
                output = model(data)
                pcd_out1 = output[1,:,:].detach().cpu().numpy()
                pcd_out2 = output[3,:,:].detach().cpu().numpy()
                pcd_out3 = output[7,:,:].detach().cpu().numpy()
                
                
                fig = plt.figure()
                bx = fig.add_subplot(321, projection='3d')
                bx.scatter(pcd_in1[:,0],pcd_in1[:,1],pcd_in1[:,2])
                ax = fig.add_subplot(322, projection='3d')
                ax.scatter(pcd_out1[:,0],pcd_out1[:,1],pcd_out1[:,2])
                bx = fig.add_subplot(323, projection='3d')
                bx.scatter(pcd_in2[:,0],pcd_in2[:,1],pcd_in2[:,2])
                ax = fig.add_subplot(324, projection='3d')
                ax.scatter(pcd_out2[:,0],pcd_out2[:,1],pcd_out2[:,2])
                bx = fig.add_subplot(325, projection='3d')
                bx.scatter(pcd_in3[:,0],pcd_in3[:,1],pcd_in3[:,2])
                ax = fig.add_subplot(326, projection='3d')
                ax.scatter(pcd_out3[:,0],pcd_out3[:,1],pcd_out3[:,2])
                plt.savefig("1.png")
                loss = chamfer_distance_loss(data, output)
            

