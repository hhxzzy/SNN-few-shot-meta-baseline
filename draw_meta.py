import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np


expName1 = 'meta_mini-imagenet-1shot_meta-baseline-resnet12_snn_PLIF'
expName5 = expName1.replace('1shot','5shot')
expName = expName1.replace('-1shot','')


fileName1 = './few-shot-meta-baseline/save/'
fileName5 = './few-shot-meta-baseline/save/'

saveName = './results/'
saveName += expName
if not os.path.exists(saveName):
    os.mkdir(saveName)
saveName1 = saveName + '/acc.png'
saveName2 = saveName + '/loss.png'

fileName1 += expName1
fileName1 += '/log.txt'
fileName5 += expName5
fileName5 += '/log.txt'



train_loss1 = []
val_loss1 = []
test_loss1 = []
train_loss5 = []
val_loss5 = []
test_loss5 = []

train_acc1 = []
val_acc1 = []
test_acc1 = []
train_acc5 = []
val_acc5 = []
test_acc5 = []


with open(fileName1, 'r') as f:
    for line in f.readlines():        
        line = re.split("[ ,|]",line)
        if line[0] == 'epoch':
            train_loss1.append(float(line[4]))
            train_acc1.append(float(line[5]))
            test_loss1.append(float(line[8]))
            test_acc1.append(float(line[9]))
            val_loss1.append(float(line[12]))
            val_acc1.append(float(line[13]))

with open(fileName5, 'r') as f:
    for line in f.readlines():        
        line = re.split("[ ,|]",line)
        if line[0] == 'epoch':
            train_loss5.append(float(line[4]))
            train_acc5.append(float(line[5]))
            test_loss5.append(float(line[8]))
            test_acc5.append(float(line[9]))
            val_loss5.append(float(line[12]))
            val_acc5.append(float(line[13]))

plt.figure()
plt.plot(range(1,len(train_acc1)+1), train_acc1, label='1shot_train_acc')
plt.plot(range(1,len(train_acc1)+1), val_acc1, label='1shot_val_acc')
plt.plot(range(1,len(train_acc1)+1), test_acc1, label='1shot_test_acc')
plt.plot(range(1,len(train_acc1)+1), train_acc5, label='5shot_train_acc')
plt.plot(range(1,len(train_acc1)+1), val_acc5, label='5shot_val_acc')
plt.plot(range(1,len(train_acc1)+1), test_acc5, label='5shot_test_acc')
# plt.yticks(np.arange(0,1,0.1))
plt.legend()
plt.savefig(saveName1)


plt.figure()
plt.plot(range(1,len(train_acc1)+1), train_loss1, label='1shot_train_loss')
plt.plot(range(1,len(train_acc1)+1), val_loss1, label='1shot_val_loss')
plt.plot(range(1,len(train_acc1)+1), test_loss1, label='1shot_test_loss')
plt.plot(range(1,len(train_acc1)+1), train_loss5, label='5shot_train_loss')
plt.plot(range(1,len(train_acc1)+1), val_loss5, label='5shot_val_loss')
plt.plot(range(1,len(train_acc1)+1), test_loss5, label='5shot_test_loss')
# plt.yticks(np.arange(0,1,0.1))
plt.legend()
plt.savefig(saveName2)
