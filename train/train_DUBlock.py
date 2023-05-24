import matplotlib.pyplot as plt
import warnings

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from models.DUBlock import TMC
from datasets.data_process import *
from utils.visualise import visual_train_DUBlock


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type = int, default = 32, metavar = 'N',
                        help = 'input batch size for training [default: 100]')
    parser.add_argument('--epochs', type = int, default = 15, metavar = 'N',
                        help = 'number of epochs to train [default: 500]')
    parser.add_argument('--lambda-epochs', type = int, default = 50, metavar = 'N',
                        help = 'gradually increase the value of lambda from 0 to 1')

    args = parser.parse_args()

    args.dims = [[512, 128, 32], [512, 128, 32]]
    args.views = len(args.dims)

    model = TMC(11, args.views, args.dims, args.lambda_epochs)
    model.cuda()

    lr = 0.003
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, 5, gamma = 0.1, last_epoch = -1) 


    # load Backbone
    cnn = torch.load('./backbone.pkl')
    cnn.cuda()
    

    label_x1 = []
    label_x2 = []
    label_y_loss = []
    label_y_accuracy = []


    def train(epoch):
        model.train()

        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        label_x2.append(epoch*len(train_loader))
        for step, (ms, pan, label, _) in enumerate(train_loader):
            label_x1.append(step + (epoch - 1)*len(train_loader))
            ms, pan, label = ms.cuda(), pan.cuda(), label.cuda()

            output, x, y = cnn(ms, pan)
            x = x.view(x.size()[0],  -1)
            y = y.view(y.size()[0],  -1)
            data = {0: x, 1: y}
            target = label
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.long().cuda())
            optimizer.zero_grad()
            evidences, evidence_a, loss, u = model(data, target, epoch)

            label_y_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            if step % 50 == 0:
                print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, loss.item(), step))
            data_num += target.size(0)
            _, predicted = torch.max(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
        print('====> acc: {:.4f}'.format(correct_num/data_num))
        label_y_accuracy.append(correct_num/data_num)
        scheduler.step() 
        


    label_x3 = []
    label_x4 = []
    label_y_loss1 = []
    label_y_accuracy1 = []


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        label_x3.append(epoch*len(test_loader))
        i = 0   
        for data, data1, target, _ in tqdm(test_loader):
            label_x3.append(i + (epoch - 1)*len(test_data))
            i += 1
            data, data1, target = data.cuda(), data1.cuda(), target.cuda()
            output, x, y = cnn(data, data1)
            x = x.view(x.size()[0],  -1)
            y = y.view(y.size()[0],  -1)

            data = {0: x, 1: y}
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            with torch.no_grad():
                target = Variable(target.long().cuda())
                evidences, evidence_a, loss, u = model(data, target, epoch)
                label_y_loss1.append(loss.item())
                _, predicted = torch.max(evidence_a.data, 1)
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())

        print('====> acc: {:.4f}'.format(correct_num/data_num))
        label_y_accuracy1.append(correct_num/data_num)
        return loss_meter.avg, correct_num/data_num





    for epoch in range(1, args.epochs + 1):
        print('train:')
        train(epoch)
        print('test:')
        test(epoch)

    torch.save(model, './DUBlock.pkl')



    visual_train_DUBlock(label_x1,
                         label_y_loss, 
                         label_x2, 
                         label_y_accuracy,
                         label_x3,
                         label_y_loss1,
                         label_x4,
                         label_y_accuracy1)






   


