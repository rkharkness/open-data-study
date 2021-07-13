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
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix, roc_curve
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from coronet_fpae_train import FPN_Gray, CustomTensorDataset
from coronet_cin_train import  Classifier

import argparse


def visualise_cam(img,pred,stacked_norm,v, model):
        pred_ind = pred.argmax(dim=1, keepdim=True).squeeze()

        if pred_ind == 0:
            label="Negative"
        elif pred_ind == 1:
            label="Pneumonia"
        elif pred_ind == 2:
            label = "COVID-19"

        pred[:, pred_ind].backward()
        gradients = model.get_gradient()

        pooled_gradients = torch.mean(gradients, dim=[0,2,3])

        activations = model.get_activations(stacked_norm).detach()

        for i in range(512):
          activations[:,i,:,:] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu()
        heatmap = np.maximum(heatmap,0)
        heatmap /= torch.max(heatmap)

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (512,512))

        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img = img.squeeze().unsqueeze(0).permute(1,2,0)

        superimposed_img = heatmap * 0.3 + (img*255).cpu().numpy()
        cv2.imwrite('/app/coronet_covidx_grad_cam_{}-{}_{}.png'.format(label,pred.cpu().detach().numpy(), v), superimposed_img.squeeze())
        
        return superimposed_img

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        img = "/app/custom_data_large_chexpert/" + self.df.filename[index]
        x = cv2.imread(img,0)

        if self.transform is not None:
          x = self.transform(image=x)["image"]

        y = torch.tensor(self.df.class_map[index])
      
        return x/255.0, y

    def __len__(self):
        return len(self.df)


parser = argparse.ArgumentParser(description='CoroNet Testing Script')
parser.add_argument('--data_csv', default='/app/open-data-study/open-data-files/balanced_covidx_data.csv', type=str, help='Path to data file')
parser.add_argument('--save_dir', default='/home/ubuntu/', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--data_dir', default='/app/open-data-study/open-data-files', type=str, help='Path to data folder')

args = parser.parse_args()

seed=0

test_df = pd.read_csv(args.data_csv)
test_df = test_df[test_df['split']=='test'] # split unstructured covidx dataframe
test_df = test_df.reset_index(drop=True)

mapping = {'normal': 0, 'non-pneumonia':0, 'pneumonia': 1, 'COVID-19': 2}

test_df['class_map'] = test_df['finding'].replace(mapping) # convert labels to numeric
print(test_df['class_map'].value_counts())

test_df['path'] = [i.split('/')[-1] for i in test_df['filename'].values]

test_transform = A.Compose(
    [ToTensorV2()]
    )

test_data = CustomTensorDataset(test_df, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=1, num_workers=0,
                         pin_memory=True, shuffle=False,
                         worker_init_fn=np.random.seed(seed))

# create alll models
model1 = FPN_Gray()
model1.to('cuda')

model2 = FPN_Gray()
model2.to('cuda')

classifier = Classifier()
classifier.to('cuda')

for k in np.arange(1,4):
    ## === Loading the pretrained FPAE ===
    model1.load_state_dict(torch.load('/app/open-data-study/weights/coronet_weights/fpae_model_num_workers_label0_'+str(k)+".pth"))
    model1.eval()   
    ## === Loading the pretrained FPAE ===
    model2.load_state_dict(torch.load('/app/open-data-study/weights/coronet_weights/fpae_model_num_workers_label1_'+str(k)+".pth"))
    model2.eval()   
    ## === Loading pretrained classifier ===
    classifier.load_state_dict(torch.load('/app/open-data-study/weights/coronet_weights/cin_classifier_k'+str(k)+".pth"))
    classifier.eval()

##  === Testing the Classification Model ===
    count=0
    correct = 0
    gt = []
    pr = []
    pr_val = []

    for data in test_loader:

        img, gt_label = data
        img = nn.functional.interpolate(img.to('cuda').float(), 512)
        img = img.cuda().float()
        gt_label = gt_label.cuda().long()
        # =================== forward =====================
        output_1, org, z = model1(img)
        output_2, org, z = model2(img)

        # create stack of residual images
        stacked = torch.cat((torch.abs(output_1-org), 
                                             torch.abs(output_2-org), 
                                             torch.abs(output_1 + output_2 - 2*org)), dim=1)
        pred_val = classifier(stacked) 

        # create gradcam saliency maps
        #visualise_cam(img, pred_val,stacked,count, classifier)

        # get the index of the max log-probability
        pred = pred_val.detach().argmax(dim=1, keepdim=True).squeeze()
        
        correct += pred.eq(gt_label.detach().view_as(pred)).sum().item()

        pr_val.append(pred_val.squeeze().detach().cpu().numpy())
        gt.append(gt_label.item())
        pr.append(pred.item())
        count = count+1

    print("REPORT")
    print(confusion_matrix(gt, pr))
    print(classification_report(gt , pr))
    test_report = classification_report(gt , pr, output_dict=True)

    # === Evaluation metrics ===
    target_names = ['Normal', 'Pneumonia', 'COVID-19']
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()
    for i in range(len(target_names)):
        # multi-class roc curve
        fpr[i], tpr[i], _ = roc_curve(np.array(gt),np.array(pr_val)[:,i], pos_label=i, 
                                        sample_weight=None, drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])    
        
        lw = 2
        plt.plot(fpr[i], tpr[i],
                      lw=lw, label=f'{target_names[i]} - (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'CoroNet')
        plt.legend(loc="lower right")
    
    plt.savefig(f'/app/coronet_new2_custom_roc_curve_{k}.png')
