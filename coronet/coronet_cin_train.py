"""code adapted from: https://github.com/chirag126/CoroNet"""
# import required libraries
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from coronet_fpae_train import FPN_Gray

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

## =============================
# CUSTOM DATALOADER
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
      img = "/app/open-data-study/open-data-files/data/" + self.df.split[index] + "/" +  self.df.img[index]
      x = cv2.imread(img,0)

      if self.transform is not None:
          x = self.transform(image=x)["image"]

      y = torch.tensor(self.df.class_map[index])
      
      return x/255.0, y

    def __len__(self):
        return len(self.df)


def make_generators(train_df, val_df, test_df, seed):
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_transform = A.Compose(
        [
        A.HorizontalFlip(p=0.2),
        A.Resize(512,512),
        ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
          A.Resize(512,512),
          # A.Normalize(mean=(0.507), std=(0.2779)),
          ToTensorV2()
        ]
    )

    train_data = CustomTensorDataset(train_df, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=16, num_workers=4,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    val_data = CustomTensorDataset(val_df, transform=test_transform)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=0,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    test_data =CustomTensorDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=0,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))
    return train_loader, val_loader, test_loader

def validate_classifier(model1, model2, loss_fn):
    val_correct = 0
    cnt = 0

    loss_list = []

    classifier.eval()
	#with torch.no_grad():
    for data in valid_loader:
            img, gt_label = data
            img = nn.functional.interpolate(img.to('cuda').float(), 512)
            img = img.cuda().float()
            gt_label = gt_label.cuda().long()
            # =================== forward =====================
            output_1, org, z = model1(img)
            output_2, org, z = model2(img)

            pred = classifier(torch.cat((torch.abs(output_1-org), 
                                         torch.abs(output_2-org), 
                                         torch.abs(output_1 + output_2 - 2*org)), dim=1))  
            val_loss = loss_fn(pred, gt_label)            
            pred = pred.detach().argmax(dim=1, keepdim=True).squeeze()

#            val_loss = loss_fn(pred, gt_label)

            loss_list.append(val_loss.item())
            # get the index of the max log-probability
            # pred = pred.detach().argmax(dim=1, keepdim=True).squeeze()
            val_correct += pred.eq(gt_label.detach().view_as(pred)).sum().item()

    classifier.train()
 
    return np.mean(loss_list), val_correct

def train_classifier(num_epoch, train_loader, valid_loader, model1, model2, classifier, optimizer, loss_fn, scheduler, k, patience=20):
  # Training
  epoch = 0
  best_acc = 0
  best_loss = float('inf')
  model1=model1

  epoch_losses = []
  epoch_acc = []

  val_epoch_losses = []
  val_epoch_acc = []


  for epoch in range(epoch, num_epoch):
      correct = 0
      cnt = 0

      loss_list = []
      for v, data in enumerate(train_loader):
          img, gt_label = data
          img = nn.functional.interpolate(img.to('cuda').float(), 512)
          img = img.cuda().float()
          # print(img.shape)
          gt_label = gt_label.cuda().long()
          # =================== forward =====================
          output_1, org, z = model1(img)
          output_2, org, z = model2(img)

          pred = classifier(torch.cat((torch.abs(output_1-org), 
                                       torch.abs(output_2-org), 
                                       torch.abs(output_1 + output_2 - 2*org)), dim=1))  

          loss = loss_fn(pred, gt_label)
          # get the index of the max log-probability
          loss_list.append(loss.item())

          pred = pred.detach().argmax(dim=1, keepdim=True).squeeze()
          correct += pred.eq(gt_label.detach().view_as(pred)).sum().item()

          # =================== backward ====================
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          # if v == 2:
          #   break

      if epoch % 1 == 0:
          val_loss, val_correct = validate_classifier(model1=model1, model2=model2, loss_fn=loss_fn)
          scheduler.step(val_loss)
          print('epoch [{}/{}], loss:{:.4f}, accuracy: {}/{} ({:.0f}%)'.
                format(epoch+1, num_epoch, np.mean(loss_list), correct,
                        len(train_loader.dataset),
                        100. * correct / len(train_loader.dataset)))
          print('epoch [{}/{}], val_loss:{:.4f}, val_accuracy: {}/{} ({:.0f}%)'.
                format(epoch+1, num_epoch, val_loss, val_correct,
                        len(valid_loader.dataset),
                        100. * val_correct / len(valid_loader.dataset)))

          train_acc = 100. * correct / len(train_loader.dataset)
          val_acc = 100. * val_correct / len(valid_loader.dataset)
          # Save best inference model
          epoch_losses.append(np.mean(loss_list))
          val_epoch_losses.append(val_loss)

          epoch_acc.append(train_acc)
          val_epoch_acc.append(val_acc)

          if best_loss > val_loss:
              save_classification_model(classifier,epoch,k)
              best_loss = val_loss
              no_improvement = 0
              print("no improvements in validation acc - {}/{}".format(no_improvement, patience))

          elif best_loss < val_loss:
              no_improvement = no_improvement +1
              print("no improvements in validation acc - {}/{}".format(no_improvement, patience))

          if no_improvement == patience:
            print("early stopped")

            break

  return classifier, epoch_losses, val_epoch_losses, epoch_acc, val_epoch_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_classification_model(model, epoch, k):
    torch.save(model.state_dict(), "/app/cin_classifier_k{}.pth".format(k))

class Classifier(nn.Module):

  def __init__(self):
      super(Classifier, self).__init__()

      self.num_classes = 3

      self.resnet = models.resnet18(pretrained=True)
      self.features_conv = nn.Sequential(*list(self.resnet.children())[:-2])

      self.avgpool = self.resnet.avgpool

      num_ftrs = self.resnet.fc.in_features      
      self.resnet.fc = nn.Linear(num_ftrs, self.num_classes)
      self.classifier = self.resnet.fc
      
      self.gradients = None
  
  def activations_hook(self, grad):
      self.gradients = grad

  def get_gradient(self):
      return self.gradients
  
  def get_activations(self, x):
      return self.features_conv(x)
    
  def forward(self, x):
      # print(x.shape)
      x = self.features_conv(x)
      # print(x.shape)
      if x.requires_grad:
        h = x.register_hook(self.activations_hook)

      x = self.avgpool(x)
      # print(x.shape)
      x = torch.flatten(x, start_dim=1)

      x = self.classifier(x)

      return x

def get_model_name(file_name, k, label):
    return file_name + "_" + "label" + str(label) + "_" + str(k)+'.pth'

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoroNet CIN Training Script')

    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--data_csv', default='/app/open-data-study/open-data-files/balanced_covidx_data.csv', type=str, help='Path to data file')
    parser.add_argument('--data_dir', default='/app/open-data-study/open-data-files', type=str, help='Path to data folder')
    parser.add_argument('--savefile', default='/app/coronet_cin', help='Filename for training data')
    parser.add_argument('--model1_weights', default='/app/open-data-study/weights/fpae_model_num_workers_label0', help='Filepath for model 1 AE saved weights')
    parser.add_argument('--model2_weights', default='/app/open-data-study/weights/fpae_model_num_workers_label1', help='Filepath for model 2 AE saved weights')

    args = parser.parse_args()
    
    num_epoch = 1000
    total_data = pd.read_csv(args.data_csv)
    total_data['class_map'] = total_data['finding'].map(mapping)

    train_df = total_data[total_data['split']=='train']
    train_df = train_df.reset_index(drop=True)

    test_df = total_data[total_data['split']=='test']
    test_df = test_df.reset_index(drop=True)

    classifier = Classifier().to('cuda')
    loss_fn = nn.CrossEntropyLoss().to('cuda')

    seed = 0
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    target = train_df.class_map

    fold_no = 1

    for train_idx, val_idx in kfold.split(train_df, target):
        train_kfold = train_df.iloc[train_idx]
        val_kfold = train_df.iloc[val_idx]

        train_loader, valid_loader, test_loader = make_generators(train_kfold, val_kfold, test_df, seed=seed)

            ## === Loading the pretrained binary FPAE ===
        model1 = FPN_Gray()
        model1.to('cuda')
        model1.load_state_dict(torch.load(args.model1_weights + "_" + str(fold_no) + ".pth"))
        model1.eval()  

        model2 = FPN_Gray()
        model2.to('cuda')
        model2.load_state_dict(torch.load(args.model2_weights + "_" + str(fold_no) + ".pth"))
        model2.eval()    
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        
        nll_loss = nn.CrossEntropyLoss().to('cuda') # changed to being unweighted - dataset now balanced

        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.9, patience=10, threshold=0.00001, 
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)
        classifier, epoch_losses, val_epoch_losses, epoch_acc, val_epoch_acc = train_classifier(num_epoch=num_epoch, train_loader=train_loader,valid_loader=valid_loader,
                                  model1=model1, model2=model2, classifier=classifier, optimizer=optimizer, loss_fn=nll_loss, k=fold_no, scheduler=scheduler)

        fold_no = fold_no + 1





