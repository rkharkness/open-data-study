"""code adapted from: https://github.com/chirag126/CoroNet"""

# import required libraries
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import time
import csv
import argparse

class FPN_Gray(nn.Module):
    def __init__(self):
        super(FPN_Gray, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=12)
        self.conv7 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=8)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.conv_smooth1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv_smooth2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv_smooth3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.convtrans1 = nn.ConvTranspose2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.convtrans3 = nn.ConvTranspose2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.convtrans4 = nn.ConvTranspose2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.convtrans5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.convtrans6 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.convtrans7 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x_small = x.clone()
        x_clone = x_small.clone()

        # ============ Encoder ===========
        # ====== Bottom Up Layers =====
        x = self.bn0(x_small)
        res1_x = self.conv1(x)
        x = self.relu(res1_x)
        x = self.bn1(x)
        res2_x = self.conv2(x)
        x = self.relu(res2_x)
        x = self.bn2(x)
        res3_x = self.conv3(x)
        x = self.relu(res3_x)
        x = self.bn3(x)
        _, _, H1, W1 = x.size()

        ### ======= Branch network ======
        x_d1 = self.down(x)  # 128x128
        _, _, H2, W2 = x_d1.size()
        x_d2 = self.down(x_d1)  # 64x64
        _, _, H3, W3 = x_d2.size()
        x_d3 = self.down(x_d2)  # 32x32

        ### ======= First Branch =======
        res4_x = self.conv4(x)
        x = self.relu(res4_x)
        x = self.bn4(x)
        res5_x = self.conv5(x)
        x = self.relu(res5_x)
        x = self.bn5(x)
        res6_x = self.conv6(x)
        x = self.relu(res6_x)
        x = self.bn6(x)
        res7_x = self.conv7(x)
        x = self.relu(res7_x)
        x = self.bn7(x)

        ### ======= Second Branch ========
        x_d1 = self.conv4(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn4(x_d1)
        x_d1 = self.conv5(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn5(x_d1)
        x_d1 = self.conv6(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn6(x_d1)
        x_d1 = self.conv7(x_d1)
        x_d1 = self.relu(x_d1)
        z1 = self.bn7(x_d1)
        x_d1 = self.upsample(z1, size=(H1, W1))

        ### ======= Third Branch ========
        x_d2 = self.conv4(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn4(x_d2)
        x_d2 = self.conv5(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn5(x_d2)
        x_d2 = self.conv6(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn6(x_d2)
        x_d2 = self.conv7(x_d2)
        x_d2 = self.relu(x_d2)
        z2 = self.bn7(x_d2)
        x_d2 = self.upsample(z2, size=(H2, W2))
        x_d2 = self.upsample(x_d2, size=(H1, W1))

        ### ======= Fourth Branch ========
        x_d3 = self.conv4(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn4(x_d3)
        x_d3 = self.conv5(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn5(x_d3)
        x_d3 = self.conv6(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn6(x_d3)
        x_d3 = self.conv7(x_d3)
        x_d3 = self.relu(x_d3)
        z3 = self.bn7(x_d3)
        x_d3 = self.upsample(z3, size=(H3, W3))
        x_d3 = self.upsample(x_d3, size=(H2, W2))
        x_d3 = self.upsample(x_d3, size=(H1, W1))

        ### ======= Concat maps ==========
        x = torch.cat((x, x_d1, x_d2, x_d3), 1)

        x = self.conv_smooth1(x)
        x = self.conv_smooth2(x)
        x = self.conv_smooth3(x)
       
        ### ============ Decoder ==========
        x = self.convtrans1(x)
        x = self.relu(x+res6_x)
        x = self.convtrans2(x)
        x = self.relu(x+res5_x)
        x = self.convtrans3(x)
        x = self.relu(x+res4_x)
        x = self.convtrans4(x)
        x = self.relu(x+res3_x)
        x = self.convtrans5(x)
        x = self.relu(x+res2_x)
        x = self.convtrans6(x)
        x = self.relu(x+res1_x)
        x = self.convtrans7(x)
        x = x + x_clone
        x = self.sigmoid(x)

        return x, x_small, z3

    def upsample(self, x, size):
        up = nn.Upsample(size=size, mode="bilinear")
        return up(x)

def get_model_name(k, label):
    return 'fpae_model_num_workers' + "_" + "label" + str(label) + "_" + str(k)+'.pth'


def make_generators(train_df, val_df):
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    transform = A.Compose(
        [
        ToTensorV2()
        ]
    )

    seed = 0
    train_data = CustomTensorDataset(train_df, transform=transform)
    train_loader = DataLoader(train_data, batch_size=1, num_workers=4,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    val_data = CustomTensorDataset(val_df, transform=transform)
    valid_loader = DataLoader(val_data, batch_size=1, num_workers=4,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    return train_loader, valid_loader

def validate(label, model):

    loss_list = []

    batch_img = torch.FloatTensor().to('cuda').float()
    for data in valid_loader:
        img, gt_label = data
        img = nn.functional.interpolate(img.to('cuda').float(), 512)
        gt_label = gt_label.to('cuda').long()
        
        if gt_label.item() == label:
            batch_img = torch.cat((batch_img, img), dim=0)
         
        if batch_img.shape[0] == 32:
             # ===================forward=====================
            optimizer.zero_grad()
            output, org, z = model(img)
            _, _, z1 = model(output)
            loss = mse_loss(output, org) + mse_loss(z, z1)

            batch_img = torch.FloatTensor().to('cuda').float()

            loss_list.append(loss.item())


    val_loss = np.mean(loss_list)

    return val_loss

def train(num_epoch, train_loader, model, optimizer, scheduler, mse_loss, seed, patience = 20, label=0):
    # FPAE training function for a given DataLoader
    epoch = 0
    no_improvements = 0
    best_loss = float('inf')

    val_loss_list = []
    train_loss_list = []

    for epoch in range(epoch, num_epoch):

      loss_list = []
      batch_img = torch.FloatTensor().to('cuda').float()
      for data in train_loader:
          img, gt_label = data
          img = img.to('cuda')
          img = nn.functional.interpolate(img.to('cuda').float(), 512)

          gt_label = gt_label.to('cuda').long()
          if gt_label.item() == label:
              batch_img = torch.cat((batch_img, img), dim=0)
         
          if batch_img.shape[0] == 32:
              # ===================forward=====================
              optimizer.zero_grad()
              output, org, z = model(img)
              _, _, z1 = model(output)
              loss = mse_loss(output, org) + mse_loss(z, z1)

              # ===================backward====================
              loss.backward()
              optimizer.step()
              batch_img = torch.FloatTensor().to('cuda').float()

              loss_list.append(loss.item())

      # ===================log========================
      val_loss = validate(label, model)
      scheduler.step(val_loss)

      print(optimizer.param_groups[0]['lr'])
      print('epoch [{}/{}], loss:{:.10f}'.
          format(epoch+1, num_epoch, np.mean(loss_list)))
      
      print('epoch [{}/{}], val_loss:{:.10f}'.
                format(epoch+1, num_epoch, val_loss))

      val_loss_list.append(val_loss)
      train_loss_list.append(np.mean(loss_list))
          
          # Save best inference model
      if best_loss > val_loss:
              save_best_model(model, filepath)
              best_loss = val_loss
              ind = epoch
              no_improvement = 0
              print("no improvements in validation loss - {}/{}".format(no_improvement, patience))

      elif best_loss < val_loss:
              no_improvement = no_improvement +1
              print("no improvements in validation loss - {}/{}".format(no_improvement, patience))

      if no_improvement == patience:
            print("early stopped")
            break

    return model, train_loss_list, val_loss_list

mapping = {'normal': 0,'non-pneumonia':0, 'pneumonia': 1, 'COVID-19': 2}

def save_best_model(model, filepath): 
  torch.save(model.state_dict(), filepath)


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
      img = self.df.filename[index]
      x = cv2.imread(img,0)

      if self.transform is not None:
          x = self.transform(image=x)["image"]

      y = torch.tensor(self.df.class_map[index])

      return x/255.0, y

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoroNet FPAE Training Script')
    parser.add_argument('--data_csv', default='/root/my-repo/balanced_covidx_data.csv', type=str, help='Path to data file')
    parser.add_argument('--data_dir', default='/root/my-repo/data/', type=str, help='Path to data folder')
    args = parser.parse_args()

    total_data = pd.read_csv(args.data_csv)
    total_data['pneumonia_binary'] = total_data['finding'].replace(mapping) 
    total_data['class_map'] = total_data['pneumonia_binary']

    total_data.filename = [i.split('/')[4:] for i in total_data.filename] # change according to number of dir in full path
    total_data.filename = [args.data_dir  + '/'.join(i) for i in total_data.filename]

    train_df = total_data[total_data['split']=='train']
    train_df = train_df.reset_index(drop=True)

    mse_loss = nn.MSELoss()

    seed = 0
    np.random.seed(seed)

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    target = train_df.class_map

    fold_no = 1

    # get indices for 3-fold cross validation
    for train_idx, val_idx in kfold.split(train_df, target):
        train_kfold = train_df.iloc[train_idx]
        val_kfold = train_df.iloc[val_idx]

        train_loader, valid_loader = make_generators(train_kfold, val_kfold)

        print('------------------------------------------------------------------------')

        # Training and testing the model on normal image data
        model = FPN_Gray()
        model.to('cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.9, patience=5, threshold=1e-10, 
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)
        filepath = '/app/' + get_model_name(fold_no, label=0)
        print(f'Training for fold {fold_no} ...')

        start_time = time.time()

        model1, train_loss_list, val_loss_list = train(num_epoch=1000, train_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler, seed=seed, mse_loss=mse_loss, label=0)

        print('## ===== Training finished for label 0 FPAE ===== ##')
        print("--- %s seconds ---" % (time.time() - start_time))

        print('------------------------------------------------------------------------')

         # Training and testing the model on non-COVID pneumonia data
        model = FPN_Gray()
        model.to('cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.9, patience=5, threshold=1e-10, 
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)
        filepath = '/app/' + get_model_name(fold_no, label=1)

        start_time = time.time()

        model2, train_loss_list, val_loss_list = train(num_epoch=1000, train_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler, seed=seed, mse_loss=mse_loss, label=1)
        print('## ===== Training finished for label 1 FPAE ===== ##')
        print("--- %s seconds ---" % (time.time() - start_time))


        fold_no = fold_no + 1

