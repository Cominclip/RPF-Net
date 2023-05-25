import torch
import torch.nn as nn
import torchvision
import torch.functional as F
from torch.autograd import Variable

import numpy as np
import math
from tqdm import tqdm

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
      
from datasets.data_process import *
from .Backbone import ClassNet
from .Representation import *



label_envirodata = []
ground_xy_envirodata = []
for i in range(len(representation)):
    label_envirodata.append(label_np[representation[i][0][0]][representation[i][0][1]])
    ground_xy_envirodata.append(representation[i][0])

label_envirodata = np.array(label_envirodata)
ground_xy_envirodata = np.array(ground_xy_envirodata)


label_envirodata = torch.from_numpy(label_envirodata).type(torch.LongTensor)
ground_xy_envirodata = torch.from_numpy(ground_xy_envirodata).type(torch.LongTensor)

envirodata = MyData(ms4, pan, label_envirodata, ground_xy_envirodata, Ms4_patch_size)
envirodata_loader = DataLoader(dataset = envirodata, batch_size = 1, shuffle = False, num_workers = 0)   #len(envirodata_loader) = 14631


backbone = torch.load('Backbone.pkl')
backbone.cuda()

Tmcmodel = torch.load('DUBlock.pkl')
Tmcmodel.cuda()

classnet = ClassNet().cuda()
classnet.outlayer.load_state_dict(backbone.outlayer.state_dict())

 
# inital_environment
sumu_ms = 0
sumu_pan = 0
u_all = [] # store the uncertainty value of ms and pan blocks: [{0: u_ms, 1: u_pan}, {0: u_ms, 1: u_pan},...]
all_fusion = [] # store the current fusion features 
ms_after_backbone = [] # store results obtained by MS and PAN through backbone network
pan_after_backbone = [] # It is stored and used in subsequent iterations without having to propagate the calculation again

state0 = []
count = 0

step_wrong_all = []  
step_wrong_label = []  
label_wrong = []  
location_wrong = []  

i = 0 
for step, (ms, pan, label, location) in enumerate(tqdm(envirodata_loader)):
    if label != 255:
        location = np.array(location)
        ms = ms.cuda()
        pan = pan.cuda()
        label = label.cuda()
        with torch.no_grad():
            output, x, y = backbone(ms, pan)
        x = x.view(x.size()[0],  -1)
        y = y.view(y.size()[0],  -1)
        ms_after_backbone.append(x)
        pan_after_backbone.append(y)
        fusion = (x + y)/2   
        all_fusion.append(fusion)
        data = {0: x, 1: y}
        target = label 
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        target = Variable(target.long().cuda())
        evidences, evidence_a, loss, u = Tmcmodel(data, target, 1)    
        u_all.append(u)
        re = classnet(fusion) 
        pred_y = torch.max(re, 1)[1].cuda().data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        label = label.cpu().numpy()
        if pred_y_numpy == label[0]:
            state0.append(1)
        else:
            state0.append(0)
            label_wrong.append(label[0])
            location_wrong.append(list(location[0]))
            step_wrong_all.append(step)
            step_wrong_label.append(i)
            count += 1
            sumu_ms += 1 - u[0]
            sumu_pan += 1 - u[1]
        i += 1  
    else:
        state0.append(1)


np.save('label_wrong.npy', label_wrong)
np.save('location_wrong.npy', location_wrong)

label_wrong = np.array(label_wrong)
location_wrong = np.array(location_wrong)


label_wrong = torch.from_numpy(label_wrong).type(torch.LongTensor)
location_wrong = torch.from_numpy(location_wrong).type(torch.LongTensor)

# Generate the environment dataset
wrong_pixdata = MyData(ms4, pan, label_wrong, location_wrong, Ms4_patch_size)
wrong_pixdata_loader = DataLoader(dataset = wrong_pixdata, batch_size = 1, shuffle = False, num_workers = 0)   #len(wrong_pixdata_loader) = 32



print('sumu_ms:', sumu_ms)   
print('sumu_pan:', sumu_pan) 



            
        






