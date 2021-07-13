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

from coronet_fpae_train import FPN_Gray, CustomTensorDataset, make_generators

import argparse

def validate_classifier(model1, model2, loss_fn):
  """evaluate model oo validation data"""
    val_correct = 0
    cnt = 0

    loss_list = []

    classifier.eval()
    for data in valid_loader:
            img, gt_label = data
            img = nn.functional.interpolate(img.to('cuda').float(), 512)
            img = img.cuda().float()
            gt_label = gt_label.cuda().long()


            # =================== forward =====================
            output_1, org, z = model1(img)
            output_2, org, z = model2(img)
            
            # make prediction based on stack of residual images
            pred = classifier(torch.cat((torch.abs(output_1-org), 
                                         torch.abs(output_2-org), 
                                         torch.abs(output_1 + output_2 - 2*org)), dim=1))  
            val_loss = loss_fn(pred, gt_label)            
            pred = pred.detach().argmax(dim=1, keepdim=True).squeeze() # get the index of the max log-probability

            loss_list.append(val_loss.item())
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
      for data in train_loader:
          img, gt_label = data
          img = nn.functional.interpolate(img.to('cuda').float(), 512)

          img = img.cuda().float()
          gt_label = gt_label.cuda().long()

          # =================== forward =====================
          output_1, org, z = model1(img) 
          output_2, org, z = model2(img)

          # predict outcome using stack of 3 residual images 
          pred = classifier(torch.cat((torch.abs(output_1-org), 
                                       torch.abs(output_2-org), 
                                       torch.abs(output_1 + output_2 - 2*org)), dim=1)) 

          loss = loss_fn(pred, gt_label)
          loss_list.append(loss.item()) # collect training losses

          # get the index of the max log-probability
          pred = pred.detach().argmax(dim=1, keepdim=True).squeeze()
          correct += pred.eq(gt_label.detach().view_as(pred)).sum().item()

          # =================== backward ====================
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()



      if epoch % 1 == 0:
          val_loss, val_correct = validate_classifier(model1=model1, model2=model2, loss_fn=loss_fn) # evaluate on val data
          scheduler.step(val_loss)

          # to monitor training and val metrics
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
              save_classification_model(classifier,epoch,k) # save model
              best_loss = val_loss
              no_improvement = 0
              print("no improvements in validation acc - {}/{}".format(no_improvement, patience))

          elif best_loss < val_loss:
              no_improvement = no_improvement +1
              print("no improvements in validation acc - {}/{}".format(no_improvement, patience))

          if no_improvement == patience: # if val loss does not improve within specified number of epochs, end training
            print("early stopped")

            break

  return classifier, epoch_losses, val_epoch_losses, epoch_acc, val_epoch_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_classification_model(model, epoch, k):
    torch.save(model.state_dict(), "/app/cin_classifier_k{}.pth".format(k))

class Classifier(nn.Module):
"""ResNet18 classifier, with gradient hooks for grad-CAM visualisations"""
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
      x = self.features_conv(x)
      if x.requires_grad:
        h = x.register_hook(self.activations_hook)

      x = self.avgpool(x)
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
    total_data['class_map'] = total_data['finding'].map(mapping) # convet labels to numeric

    train_df = total_data[total_data['split']=='train']
    train_df = train_df.reset_index(drop=True)

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

        train_loader, valid_loader = make_generators(train_kfold, val_kfold, seed=seed)

        ## === Loading the pretrained FPAE ===
        model1 = FPN_Gray()
        model1.to('cuda')
        model1.load_state_dict(torch.load(args.model1_weights + "_" + str(fold_no) + ".pth"))
        model1.eval()  
        ## === Loading the pretrained FPAE ===
        model2 = FPN_Gray()
        model2.to('cuda')
        model2.load_state_dict(torch.load(args.model2_weights + "_" + str(fold_no) + ".pth"))
        model2.eval()    
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        
        nll_loss = nn.CrossEntropyLoss().to('cuda') # changed to being unweighted - dataset now balanced

        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5, weight_decay=1e-3)
        # reduce lr on loss plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.9, patience=10, threshold=0.00001, 
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)
        classifier, epoch_losses, val_epoch_losses, epoch_acc, val_epoch_acc = train_classifier(num_epoch=num_epoch, train_loader=train_loader,valid_loader=valid_loader,
                                  model1=model1, model2=model2, classifier=classifier, optimizer=optimizer, loss_fn=nll_loss, k=fold_no, scheduler=scheduler)

        fold_no = fold_no + 1





