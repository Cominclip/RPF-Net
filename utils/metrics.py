import numpy as np
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from datasets.data_process import *

ALL = len(label_test)

results = np.load('results.npy', allow_pickle = True)

acc = 0
for i in range(Categories_Number):
    count = 0
    for j in range(len(ground_xy[i])):
        if results[ground_xy[i][j][0]][ground_xy[i][j][1]] == 0:
            count += 1
    print('Accuracy of catrgory %d is %f'%(i, (len(ground_xy[i]) - count)/len(ground_xy[i])))
    acc += (len(ground_xy[i]) - count)/len(ground_xy[i])

# OA
count = 0
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        if results[i][j] == 0:
            count += 1
print('OA:', (ALL - count)/ALL)


# AA
print('AA:', acc/Categories_Number)


# Kappa
from sklearn.metrics import cohen_kappa_score

y_true = []
y_pred = []

location_of_test = np.load('location_test.npy', allow_pickle = True)
results_detail = np.load('results_detail.npy', allow_pickle = True)

for i in range(len(location_of_test)):
    y_true.append(label_np[location_of_test[i][0]][location_of_test[i][1]].item())
    y_pred.append(results_detail[location_of_test[i][0]][location_of_test[i][1]].item())
kappa_value = cohen_kappa_score(y_true, y_pred)
print("Kappa:", kappa_value)






















