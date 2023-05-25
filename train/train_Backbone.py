import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from models.Backbone import *
from datasets.data_process import *
from utils.visualise import visual_train_Backbone


model = Backbone().cuda()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 13, gamma=0.1, last_epoch=-1)


label_x1 = []
label_x2 = []
label_y_loss = []
label_y_accuracy = []


def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    for step, (ms, pan, label, _) in enumerate(tqdm(train_loader)):
        label_x1.append(step + (epoch - 1)*len(train_loader))
        ms, pan, label = ms.cuda(), pan.cuda(), label.cuda()

        optimizer.zero_grad()
        output, x, y = model(ms, pan)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        loss = F.cross_entropy(output, label.long())
        label_y_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, loss.item(), step))
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))
    scheduler.step() # Update learning rate



def test_model(model, test_loader, epoch):
    label_x2.append(epoch)
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, data1, target, _ in tqdm(test_loader):
            data, data1, target = data.cuda(), data1.cuda(), target.cuda()

            output, x, y = model(data, data1)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()

        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))
        label_y_accuracy.append(100.0 * correct / len(test_loader.dataset))


for epoch in range(1, EPOCH + 1):
    train_model(model, train_loader, optimizer, epoch)
    test_model(model, test_loader, epoch)

torch.save(model, './Backbone.pkl')

visual_train_Backbone(label_x1, label_y_loss, label_x2, label_y_accuracy)
