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

#        img = cv2.resize(img.cpu().numpy(), (256, 256))

        heatmap = cv2.resize(heatmap, (512,512))
        # print(heatmap.shape)
        # print(img.shape)
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # img = (img*255).squeeze().unsqueeze(0)
        # img = torch.cat((img,img,img),dim=2)
        # print(heatmap.shape)
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

parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--img_size', default=512, type=int, help='Image size')
parser.add_argument('--img_channel', default=1, type=int, help='Image channel')
parser.add_argument('--data_csv', default='/app/open-data-study/open-data-files/balanced_covidx_data.csv', type=str, help='Path to data file')
parser.add_argument('--save_dir', default='/home/ubuntu/', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--data_dir', default='/app/open-data-study/open-data-files', type=str, help='Path to data folder')
parser.add_argument('--savefile', default='/app/', help='Filename for training data')
parser.add_argument('--ae_weightsfile', default='coronet_fpae', help='Filepath for AE saved weights')
parser.add_argument('--cin_weightsfile', default='coronet_fpae', help='Filepath for CIN saved weights')

args = parser.parse_args()

seed=0

test_df = pd.read_csv('/app/custom_data_large_chexpert.csv')
#test_df = test_df[test_df['split']=='test'] # split unstructured covidx dataframe
test_df = test_df.reset_index(drop=True)

mapping = {'normal': 0, 'non-pneumonia':0, 'pneumonia': 1, 'COVID-19': 2}
test_df['class_map'] = test_df['finding'].replace(mapping)
print(test_df['class_map'].value_counts())

test_df['path'] = [i.split('/')[-1] for i in test_df['filename'].values]

test_transform = A.Compose(
    [
          # A.Normalize(mean=(0.507), std=(0.2779)),
          ToTensorV2()
    ]
    )

test_data = CustomTensorDataset(test_df, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=1, num_workers=0,
                         pin_memory=True, shuffle=False,
                         worker_init_fn=np.random.seed(seed))



model1 = FPN_Gray()
model1.to('cuda')

model2 = FPN_Gray()
model2.to('cuda')

classifier = Classifier()
classifier.to('cuda')

for k in np.arange(1,4):
    ## === Loading the pretrained binary FPAE ===

    model1.load_state_dict(torch.load('/app/open-data-study/weights/coronet_weights/fpae_model_num_workers_label0_'+str(k)+".pth"))
    model1.eval()   
    
    model2.load_state_dict(torch.load('/app/open-data-study/weights/coronet_weights/fpae_model_num_workers_label1_'+str(k)+".pth"))
    model2.eval()   
    ## === Loading model1 weights ===
    classifier.load_state_dict(torch.load('/app/open-data-study/weights/coronet_weights/cin_classifier_k'+str(k)+".pth"))
    classifier.eval()

##  === Testing the Classification Model ===
    count=0
    correct = 0
    gt = []
    pr = []
    pr_val = []
#    with torch.no_grad():
    for data in test_loader:

        img, gt_label = data
        img = nn.functional.interpolate(img.to('cuda').float(), 512)
        img = img.cuda().float()
        gt_label = gt_label.cuda().long()
        # =================== forward =====================
        output_1, org, z = model1(img)
        output_2, org, z = model2(img)
        stacked = torch.cat((torch.abs(output_1-org), 
                                             torch.abs(output_2-org), 
                                             torch.abs(output_1 + output_2 - 2*org)), dim=1)
        pred_val = classifier(stacked) 
        #visualise_cam(img, pred_val,stacked,count, classifier)
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
# print(test_report)

# ## === Evaluation metrics ###
# mapping = ['negative', 'positive']
# recall = recall_score(gt, pr, average='weighted')
# class_wise_recall = recall_score(gt, pr, average=None)
# print(f'Sensitivity of each class:\n{mapping[0]} = {class_wise_recall[0]:.4f} | {mapping[1]} = {class_wise_recall[1]:.4f}\n')

# precision = precision_score(gt, pr, average='weighted')
# class_wise_precision = precision_score(gt, pr, average=None)
# print(f'PPV of each class:\n{mapping[0]} = {class_wise_precision[0]:.4f} | {mapping[1]} = {class_wise_precision[1]:.4f}\n')
    target_names = ['Normal', 'Pneumonia', 'COVID-19']
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()
    for i in range(len(target_names)):
#        print(gt[:,i])
 #       print(pr_val[:,i])
        #fpr[i], tpr[i], _ = roc_curve(np.array(gt).reshape(-1,1)[:,i], np.array(pr_val).reshape(-1,1)[:,i], pos_label=i, 
       #                                 sample_weight=None, drop_intermediate=False)
        fpr[i], tpr[i], _ = roc_curve(np.array(gt),np.array(pr_val)[:,i], pos_label=i, 
                                        sample_weight=None, drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])    
        
    #    plt.figure()
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
