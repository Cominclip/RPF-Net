import numpy as np
import os
from libtiff import TIFF
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .dataset import MyData, MyData_for_all

EPOCH = 45   
BATCH_SIZE = 32 
LR = 0.001  
Train_Rate = 0.01 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   


ms4_tif = TIFF.open('./data/image6/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()

size = np.shape(ms4_np)

pan_tif = TIFF.open('./data/image6/pan.tif', mode='r')
pan_np = pan_tif.read_image()

label_np = np.load("./data/image6/label6.npy")


Ms4_patch_size = 16  
Interpolation = cv2.BORDER_REFLECT_101 

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)


Pan_patch_size = Ms4_patch_size * 4  
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)


label_np = label_np - 1  
label_element, element_count = np.unique(label_np, return_counts=True)  

Categories_Number = len(label_element) - 1 
label_row, label_column = np.shape(label_np)

def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy = np.array([[]] * Categories_Number).tolist()   
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2) 

count = 0
for row in range(label_row): 
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]  
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])    

for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []
ground_xy_all = []
label_all = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    for i in range(categories_number):
        ground_xy_all.append(ground_xy[categories][i])
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_all = label_all + [categories for x in range(categories_number)]
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)
ground_xy_all = np.array(ground_xy_all)
label_all = np.array(label_all)


shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)
label_all = torch.from_numpy(label_all).type(torch.LongTensor)
ground_xy_all = torch.from_numpy(ground_xy_all).type(torch.LongTensor)


ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0) 
ms4 = np.array(ms4).transpose((2, 0, 1)) 


ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData_for_all(ms4, pan, ground_xy_allData, Ms4_patch_size)
all = MyData(ms4, pan, label_all, ground_xy_all, Ms4_patch_size) 


train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
all_loader = DataLoader(dataset = all, batch_size = 1, shuffle = False, num_workers = 0) 



