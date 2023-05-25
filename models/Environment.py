import torch
from torch.autograd import Variable

import numpy as np
import math
from tqdm import tqdm

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from .Backbone import ClassNet
from .initialize import *


# load model
backbone = torch.load('Backbone.pkl')
backbone.cuda()

Tmcmodel = torch.load('DUBlock.pkl')
Tmcmodel.cuda()

classnet = ClassNet().cuda()
classnet.outlayer.load_state_dict(backbone.outlayer.state_dict())

def getaction(output):
    if output == 0:
        k1, k2 = -1, -1
    elif output == 1:
        k1, k2 = -1, 1
    elif output == 2:
        k1, k2 = 1, -1
    elif output == 3:
        k1, k2 = 1, 1
    return k1, k2

def getreward(s):
    count = 0
    for i in range(len(s)):
        if s[i] == 0:
            count += 1
    reward = -math.exp(1 - (count/50)) * math.log(count/len(representation)) - 1
    return reward


zero = np.zeros(512)
zero = np.expand_dims(zero, axis=0) 
zero = torch.from_numpy(zero).type(torch.FloatTensor)


def getstate(k1, k2, sumu_fu, all_fusion, times, data_loader):
    if times == 1:
        alpha = 1/3
        beta = 1/3
    else:
        alpha = (sumu_ms/sumu_fu) * (1/(times + 2))
        beta = (sumu_pan/sumu_fu) * (1/(times + 2))
    s_ = np.ones(14631) 
    sumu_fu_ = 1  
    all_fusion_ = all_fusion  
    count_ = 0 
    for step, (ms, pan, label, location) in enumerate(data_loader):
        label = label.cuda()
        x = ms_after_backbone[step_wrong_label[step]]
        y = pan_after_backbone[step_wrong_label[step]]
        pre_fusion = all_fusion[step_wrong_label[step]]
        data = {0: pre_fusion, 1: zero}
        target = label
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        target = Variable(target.long().cuda())
        evidences, evidence_a, loss, u = Tmcmodel(data, target, 1) 
        fusion = (1/(1 + alpha + beta))*pre_fusion + ((alpha + beta)/(1 + alpha + beta))*(((1 - u_all[step_wrong_label[step]][0])/(2 - u_all[step_wrong_label[step]][0] - u_all[step_wrong_label[step]][1]))*k1*x + ((1 - u_all[step_wrong_label[step]][1])/(2 - u_all[step_wrong_label[step]][0] - u_all[step_wrong_label[step]][1]))*k2*y) 
        all_fusion_[step_wrong_label[step]] = fusion
        sumu_fu_ += 1 - u[0]  
        re = classnet(fusion)
        pred_y = torch.max(re, 1)[1].cuda().data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        label = label.cpu().numpy()
        if pred_y_numpy != label[0]:
            s_[step_wrong_all[step]] = 0
            count_ += 1
    s_ = list(s_)
    r = getreward(s_)
    if times >= 10:
        done = True
    else:
        done = False
    return s_, r, done, sumu_fu_, all_fusion_, count_




 


