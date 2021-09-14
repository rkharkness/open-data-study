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

from sklearn.utils import class_weight

import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



class CustomTensorDataset(Dataset):

	"""TensorDataset with support of transforms."""

	def __init__(self, df, transform=None):
		self.df = df
		self.transform = transform

	def __getitem__(self, index):
		img = os.path.join('/home/ubuntu/frankenstein_data', self.df.split[index], self.df.img[index])
		x = cv2.imread(img)

		x = self.transform(image=x)["image"]
		x = x/255.0

		x = covid
		y = torch.tensor(self.df.source_map[index])

		return x/255.0, y

	def __len__(self):
		return len(self.df)

def make_generators(train_df, val_df, test_df, seed):
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    transform = A.Compose(
        [
          A.Resize(480,480),
          ToTensorV2()
        ]
    )

    train_data = CustomTensorDataset(train_df, transform=transform)
    train_loader = DataLoader(train_data, batch_size=24, num_workers=2,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    val_data = CustomTensorDataset(val_df, transform=transform)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=0,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    test_data =CustomTensorDataset(test_df, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=0,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    return train_loader, val_loader, test_loader


class DCNN(nn.Module):
	"""ResNet18-based deep convolutinal neural network, with gradient hooks for gradCAM visualisation"""
	def __init__(self):
		super(DCNN, self).__init__()
		self.num_classes = 4

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


def validate_classifier(loss_fn, classifier):
    val_correct = 0
    cnt = 0

    loss_list = []

    classifier.eval()
	#with torch.no_grad():
    for data in valid_loader:
            img, gt_label = data
            img = img.cuda().float()
            gt_label = gt_label.cuda().long()

            # =================== forward =====================
            pred = classifier(img)

            val_loss = loss_fn(pred, gt_label)            
            pred = pred.detach().argmax(dim=1, keepdim=True).squeeze()

            loss_list.append(val_loss.item())
            val_correct += pred.eq(gt_label.detach().view_as(pred)).sum().item()

    classifier.train()
 
    return np.mean(loss_list), val_correct

def save_classification_model(model, epoch, k):
    torch.save(model.state_dict(), "/home/ubuntu/frankenstein_dcnn_k-{}.pth".format(k))

def train_classifier(num_epoch, train_loader, valid_loader, classifier, optimizer, loss_fn, scheduler, k, patience=10):
	# Training
	epoch = 0
	best_acc = 0
	best_loss = float('inf')

	epoch_losses = []
	epoch_acc = []

	val_epoch_losses = []
	val_epoch_acc = []


	for epoch in range(epoch, num_epoch):
	  correct = 0
	  cnt = 0

	  loss_list = []
	  for img, gt_label in train_loader:
	      img = img.cuda().float()
	      gt_label = gt_label.cuda().long()

	      # =================== forward =====================
	      pred = classifier(img)  
	      loss = loss_fn(pred, gt_label)

	      loss_list.append(loss.item())

	      # get the index of the max log-probability
	      pred = pred.detach().argmax(dim=1, keepdim=True).squeeze()
	      correct += pred.eq(gt_label.detach().view_as(pred)).sum().item()

	      # =================== backward ====================
	      optimizer.zero_grad()
	      loss.backward()
	      optimizer.step()

	  if epoch % 1 == 0:
	      val_loss, val_correct = validate_classifier(loss_fn=loss_fn, classifier=classifier)
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
	        save_classification_model(classifier, epoch, k)
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

if __name__ == '__main__':
	
	num_epoch = 1000 

	total_data = pd.read_csv('/MULTIX/DATA/HOME/frankenstein_data.csv')

	mapping = {'rsna':0, 'sirm':1, 'ricord':2,'cohen':3}
	total_data['source_map'] = total_data['source'].map(mapping)

	print(total_data['source_map'].value_counts())
	train_df = total_data[total_data['source_split']=='train']
	test_df = total_data[total_data['source_split']=='test']

	seed = 0
	np.random.seed(seed)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
	target = train_df.source_map

	# compute class weights to account for class imbalances  in data
	class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(target),
                                                 target)

	fold_no = 1
	for train_idx, val_idx in kfold.split(train_df, target):
		# select indices for 3 fold cross validation
		train_kfold = train_df.iloc[train_idx]
		val_kfold = train_df.iloc[val_idx]

		train_loader, valid_loader, test_loader = make_generators(train_kfold, val_kfold, test_df, seed=seed)

            ## === Loading the pretrained binary FPAE ===
		classifier = DCNN().to('cuda')

		print('------------------------------------------------------------------------')
		print(f'Training for fold {fold_no} ...')
        
		nll_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction='sum').to('cuda') # changed to being unweighted - dataset now balanced
		optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.9, patience=5, threshold=0.00001, 
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)
		classifier, epoch_losses, val_epoch_losses, epoch_acc, val_epoch_acc = train_classifier(num_epoch=num_epoch, train_loader=train_loader,valid_loader=valid_loader,
        	classifier=classifier, optimizer=optimizer, loss_fn=nll_loss, k=fold_no, scheduler=scheduler)

		fold_no = fold_no + 1


