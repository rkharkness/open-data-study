"""code adapted from: https://github.com/muhammedtalo/COVID-19"""
import warnings
warnings.filterwarnings("ignore")

from fastai.vision import *
from fastai.imports import *
from fastai.layers import *
import pandas as pd
import numpy as np
import time

import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

path = Path('/app/open-data-study/open-data-files/balanced_structured_covidx_data/train')
print(path)

np.random.seed(41)

class SaveBestModel(Recorder):
    def __init__(self, learn, name='best_model'):
        super().__init__(learn)
        self.name = name
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.save_when_acc
        
    def save_when_acc(self, metrics):        
        loss, acc = metrics[0], metrics[1]
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.learn.save(f'{self.name}')
            print("Save the best accuracy {:.5f}".format(self.best_acc))
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.learn.save(f'{self.name}')
            print("Accuracy is eq, Save the lower loss {:.5f}".format(self.best_loss))
            
    def on_epoch_end(self,last_metrics=MetricsList,**kwargs:Any):
        self.save_method(last_metrics)

def conv_block(ni, nf, size=3, stride=1):
    for_pad = lambda s: s if s > 2 else 3
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=size, stride=stride,
                  padding=(for_pad(size) - 1)//2, bias=False), 
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)  
    )

def triple_conv(ni, nf):
    return nn.Sequential(
        conv_block(ni, nf),
        conv_block(nf, ni, size=1),  
        conv_block(ni, nf)
    )

def maxpooling():
    return nn.MaxPool2d(2, stride=2)


darkcovidnet = nn.Sequential(
    conv_block(3, 8),
    maxpooling(),
    conv_block(8, 16),
    maxpooling(),
    triple_conv(16, 32),
    maxpooling(),
    triple_conv(32, 64),
    maxpooling(),
    triple_conv(64, 128),
    maxpooling(),
    triple_conv(128, 256),
    conv_block(256, 128, size=1),
    conv_block(128, 256),
    conv_layer(256, 3),
    Flatten(),
    nn.Linear(507, 3)
)

class Model(nn.Module):
    def __init__(self, darkcovidnet):
        super(Model, self).__init__()
        self.darkcovidnet = darkcovidnet
        self.features_conv = self.darkcovidnet[:-2]
        self.fc = self.darkcovidnet[-2:]

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        if x.requires_grad:
            h = x.register_hook(self.activations_hook)

        x = self.fc(x)

        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

model = Model(darkcovidnet)

kf = KFold(n_splits=3)
skf=StratifiedKFold(n_splits=3)

data=(ImageList.from_folder(path)
.split_none()
.label_from_folder()
.transform(size=(256,256))
.databunch()).normalize(imagenet_stats)

df=data.to_df()

# +
k=1
training_times_100_epochs = []
for train_index, test_index in skf.split(df.index, df['y']):
    print(f'Training for fold {k} ...')
    start_time = time.time()
    print(len(train_index), len(test_index))

    print((train_index), (test_index))

    d = (ImageList.from_folder (path)
    .split_by_idxs(train_index, test_index)
    .label_from_folder()
    .transform(size = (256,256))
    .databunch(num_workers =8)).normalize(imagenet_stats)

    learn = Learner(d, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
    learn.model.cuda()

    callbacks = [
        # EarlyStoppingCallback(learn, min_delta=1e-5, patience=10),
        SaveBestModel(learn, name=f'/app/darkcovidnet_results/darkcovidnet_model_nw8_{k}')
    ]

    learn.callbacks = callbacks
    learn.fit_one_cycle(100, max_lr=3e-3)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    training_times_100_epochs.append(time.time() - start_time)
    k = k + 1

with open('/app/darkcovidnet-100epoch-times.csv', 'w', newline='') as timefile:
    wr = csv.writer(timefile, quoting=csv.QUOTE_ALL)
    wr.writerow(training_times_100_epochs)
# -


