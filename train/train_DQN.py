import torch                                    
import torch.nn as nn                           
import torch.nn.functional as F                 
import numpy as np                              
import gym                                      
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   


# from RLModule.Pre_Environment import wrong_pixdata_loader
from models.Environment import *
from datasets.data_process import *
from models.initialize import *
from models.DQN import *
from utils.visualise import visual_train_DQN



label_wrong = np.array(label_wrong)
location_wrong = np.array(location_wrong)

label_wrong = torch.from_numpy(label_wrong).type(torch.LongTensor)
location_wrong = torch.from_numpy(location_wrong).type(torch.LongTensor)


wrong_pixdata = MyData(ms4, pan, label_wrong, location_wrong, Ms4_patch_size)
wrong_pixdata_loader = DataLoader(dataset = wrong_pixdata, batch_size = 1, shuffle = False, num_workers = 0)   #len(wrong_pixdata_loader) = 32


dqn = DQN()                                                        

all_fusion0 = all_fusion
label_x = []
label_y_reward = []
data_loader = wrong_pixdata_loader

for i in tqdm(range(2000)):                                        
    print('<<<<<<<<<Episode: %s' % i)
    s = state0                                                          
    times = 1                                                          
    episode_reward_sum = 0                                              
    sumu_fu = 0
    all_fusion = all_fusion0
    count = 32                                                        

    while True:                                                        
        a = dqn.choose_action(s)                                        
        k1, k2 = getaction(a)
        s_, r, done, sumu_fu_, all_fusion_, count_ = getstate(k1, k2, sumu_fu, all_fusion, times, data_loader)            
        times += 1
        print(' action:%s   wrong number:%s'%( a, count_))
        dqn.store_transition(s, a, r, s_)                
        episode_reward_sum += r                           

        # update state
        s = s_                                               
        sumu_fu = sumu_fu_
        all_fusion = all_fusion_

        if dqn.memory_counter > MEMORY_CAPACITY:              
            dqn.learn()
        if done:       
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            label_y_reward.append(episode_reward_sum)
            break                                             


torch.save(dqn, './DQN.pth')
visual_train_DQN(label_x, label_y_reward)
