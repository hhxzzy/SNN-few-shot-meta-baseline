import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np

expName = 'classifier_mini-imagenet_resnet12_snn_PLIF'



fileName = './few-shot-meta-baseline/save/'
saveName = './results/'

saveName += expName
if not os.path.exists(saveName):
    os.mkdir(saveName)
saveName1 = saveName + '/acc.png'
saveName2 = saveName + '/loss.png'
saveName3 = saveName + '/fsa.png'
fileName += expName
fileName += '/log.txt'



train_loss = []
val_loss = []

train_acc = []
val_acc = []

fsa_1 = []
fsa_5 = []

with open(fileName, 'r') as f:
    for line in f.readlines():        
        line = re.split("[ ,|]",line)
        if line[0] == 'epoch':
            train_loss.append(float(line[4]))
            train_acc.append(float(line[5]))
            val_loss.append(float(line[8]))
            val_acc.append(float(line[9]))
            if len(line)>13:            
                fsa_1.append(float(line[13]))
                fsa_5.append(float(line[15]))

plt.figure()
plt.plot(range(1,len(train_acc)+1), train_acc, label='train_acc')
plt.plot(range(1,len(train_acc)+1), val_acc, label='val_acc')
# plt.yticks(np.arange(0,1,0.1))
plt.legend()
plt.savefig(saveName1)


plt.figure()
plt.plot(range(1,len(train_acc)+1), train_loss, label='train_loss')
plt.plot(range(1,len(train_acc)+1), val_loss, label='val_loss')
# plt.yticks(np.arange(0,1,0.1))
plt.legend()
plt.savefig(saveName2)


plt.figure()
plt.plot(range(1,len(train_acc)+1,5), fsa_1, label='1-shot')
plt.plot(range(1,len(train_acc)+1,5), fsa_5, label='5-shot')
# plt.yticks(np.arange(0,1,0.1))
plt.legend()
plt.savefig(saveName3)