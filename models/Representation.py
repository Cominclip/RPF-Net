from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from tqdm import tqdm

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from datasets.data_process import *


image = cv2.imread('./data/image6/hxtms4.png')
segments = slic(img_as_float(image), n_segments=10000, sigma=10)
 
# show the output of SLIC
fig = plt.figure('Superpixels')
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()


segments = np.array(segments) 
print("np.unique(segments):", np.unique(segments))  

patch_num = len(np.unique(segments))  
patch = [] 
representation = []  

for i in range(patch_num):
    patch.append([])

for i in range(size[0]):
    for j in range(size[1]):
        patch[segments[i][j] - 1].append([i, j])

for i in tqdm(range(patch_num)):
    count = 0
    location = [] 
    for j in range(len(patch[i])):
        if label_np[patch[i][j][0]][patch[i][j][1]] != 255:
            count += 1
            location.append([patch[i][j][0], patch[i][j][1]])
    if count == 0:   
        representation.append([patch[i][0], 0])  
    else:
        choice = random.randint(0, count - 1)
        representation.append([location[choice], 1])




