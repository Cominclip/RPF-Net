import numpy as np
import matplotlib.pyplot as plt


def visual_train_Backbone(label_x1,
                          label_y_loss, 
                          label_x2, 
                          label_y_accuracy):
    fig = plt.figure()
    plt.plot(label_x1, label_y_loss, color='blue')
    plt.legend(['train_loss'], loc='upper right')
    plt.xlabel('number of training examples ')
    plt.ylabel('loss')
    plt.show()

    plt.plot(label_x2, label_y_accuracy, color='red')
    plt.legend(['test_accuracy'], loc='upper right')
    plt.xlabel('number of testing examples ')
    plt.ylabel('accuracy')
    plt.show()


def visual_train_DUBlock(label_x1,
                         label_y_loss, 
                         label_x2, 
                         label_y_accuracy,
                         label_x3,
                         label_y_loss1,
                         label_x4,
                         label_y_accuracy1):
    fig, ax1 = plt.subplots()
    ax1.plot(label_x1, label_y_loss, color = 'blue', label = 'loss')
    ax1.set_xlabel('number of training examples')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(label_x2, label_y_accuracy, color = 'red', label = 'accuracy')
    ax2.set_ylabel('accuracy')
    fig.legend(loc = 'upper right', bbox_to_anchor  = (1, 1), bbox_transform = ax1.transAxes)
    plt.show()

    fig, ax3 = plt.subplots()
    ax3.plot(label_x3, label_y_loss1, color = 'blue', label = 'loss')
    ax3.set_xlabel('number of testing examples')
    ax3.set_ylabel('loss')

    ax4 = ax3.twinx()
    ax4.plot(label_x4, label_y_accuracy1, color = 'red', label = 'accuracy')
    ax4.set_ylabel('accuracy')
    fig.legend(loc = 'upper right', bbox_to_anchor  = (1, 1), bbox_transform = ax1.transAxes)
    plt.show()


def visual_train_DQN(label_x, 
                     label_y_reward):
    fig = plt.figure()
    label_x = [i for i in range(2000)]
    plt.plot(label_x, label_y_reward, color='blue')
    plt.legend(['reward'], loc='upper right')
    plt.xlabel('eposide ')
    plt.ylabel('reward')
    plt.show()
