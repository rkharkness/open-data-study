import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms

from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix, classification_report, roc_curve
from sklearn.utils import class_weight

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from  torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2

from lung_segmentation.models import ResNetUNet, VGGUNet, VGGNestedUNet
from lung_segmentation.seg_tools import reverse_transform, masks_to_colorimg, plot_side_by_side, get_box_from_mask, visualize_bbox, plot_img_array, bbox

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from functools import reduce
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from tqdm import tqdm

from matplotlib.path import Path


from covidx_frankenstein import DCNN

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

		y = torch.tensor(self.df.source_map[index])

		return x, y

	def __len__(self):
		return len(self.df)

def make_generators(train_df, test_df, seed):
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    transform = A.Compose(
        [
          A.Resize(480,480),
          ToTensorV2()
        ]
    )

    train_data = CustomTensorDataset(train_df,  transform=transform)
    train_loader = DataLoader(train_data, batch_size=24, num_workers=0,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))


    test_data =CustomTensorDataset(test_df,transform=transform)
    test_loader = DataLoader(test_data, batch_size=12, num_workers=0,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))

    return train_loader, test_loader

def visualise_cam(img,pred,v,model):
	"""Create grad-CAM saliency maps """
	pred_ind = pred.argmax(dim=1, keepdim=True).squeeze()
	mapping = {0: 'RNSA', 1:'CHOWDHURY', 2:'COHEN', 3:'RICORD'}
	pred[:, pred_ind].backward()

	gradients = model.get_gradient()
	pooled_gradients = torch.mean(gradients, dim=[0,2,3])

	activations = model.get_activations(img).detach()
	for i in range(512):
		activations[:,i,:,:] *= pooled_gradients[i]

		heatmap = torch.mean(activations, dim=1).squeeze().cpu()
		heatmap = np.maximum(heatmap,0)
		heatmap /= torch.max(heatmap)

		heatmap = heatmap.numpy()

		heatmap = cv2.resize(heatmap, (480,480))

		heatmap = np.uint8(255*heatmap)
		heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
		img = img.squeeze().permute(1,2,0)
        # img = img.squeeze().unsqueeze(0).permute(1,2,0)

		superimposed_img = heatmap * 0.3 + (img*255).cpu().numpy()
		cv2.imwrite('/MULTIX/DATA/frank_gradcam/frankenstein_covidx_grad_cam_{}_{}.png'.format(pred_ind.cpu().detach().numpy(), v), superimposed_img.squeeze())
        
		return superimposed_img

def reduce_then_tsne(features, fit, pca, tsne, reduce=20, n=5000):
	print(features.shape)

	if fit==True:
		pca_results = pca.fit_transform(features)
	else:
		pca_results = pca.transform(features)

	if fit==True:
		tsne_results = tsne.fit_transform(pca_results)
	else:
		tsne_results = tsne.fit_transform(pca_results)


	return pca, tsne, pca_results, tsne_results

def covidx_seg(img): #batch
	with torch.no_grad():
		seg = seg_model(img)
		seg = seg.cpu()
		seg_box_coord = [bbox(x) for x in seg]
		print(len(seg_box_coord))

		mask = np.ones(img.shape,np.uint8)

		invert_mask = []
		for i in seg_box_coord:
			img = img.cpu()

			xmin = i[0]
			xmax = i[1]
			ymin = i[2]
			ymax = i[3]

			poly_verts = [(xmin,ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

			x, y = np.meshgrid(np.arange(480), np.arange(480))
			x, y = x.flatten(), y.flatten()

			points = np.vstack((x,y)).T

			path = Path(poly_verts)
			grid = path.contains_points(points)
			grid = grid.reshape((ny,nx))
			mask[i[0]:i[1], i[2]:i[3]] = img[i[0]:i[1], i[2]:i[3]]

			invert_seg = [1 - x for x in mask]
			invert_mask.append(invert_seg)

			img = torch.tensor(invert_seg) * img
			# plt.imshow(img[0].permute(1,2,0))
			plt.imshow(torch.tensor(img[0]*255.0).permute(1,2,0))
			plt.savefig('/MULTIX/DATA/HOME/covidx_frankenstein_seg.png')

		return img.cuda()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def tsne_vis(tsne_results, gt_labels, fit, n=5000):
	target_names = ['RSNA', 'CHOWDHURY', 'COHEN', 'RICORD']

	plt.figure(figsize=(10,10))
	plt.title("Deep CNN: t-SNE Results")

	for cl in range(4):
	    indices = np.where(gt_labels==cl)
	    indices = indices[0]
	    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1],label=target_names[cl], alpha=0.5)
	    plt.xlabel("Dimension 1")
	    plt.ylabel("Dimension 2")

	if fit == True:
		title = "/MULTIX/HOME/DATA/frankenstein_study/tsne_plot_train.png"
	else:
		title = "/MULTIX/HOME/DATA/frankenstein_study/tsne_plot_test.png"
	plt.legend()
	plt.savefig(title)
	# plt.show()


def pca_tsne(data_loader, pca, tsne, fit=True, segmentation=True):
	out_target = []
	out_output =[]
			
	with torch.no_grad():
		for data in tqdm(data_loader):
			img, gt_label = data
			img = img.cuda().float()

			if segmentation == True:
				seg = seg_model(img)
				seg = seg.cpu()
				seg_box_coord = [bbox(x) for x in seg]

				mask = np.ones(img.shape,np.uint8)

				invert_mask = []
				for i in seg_box_coord:
					img = img.cpu()
					mask[i[0]:i[1], i[2]:i[3]] = img[i[0]:i[1], i[2]:i[3]]
					invert_seg = [1 - x for x in mask]
					invert_mask.append(invert_seg)

				img = torch.tensor(invert_seg) * img
				plt.imshow(img)
				plt.savefig('/MULTIX/DATA/HOME/covidx_frankenstein_seg.png')
			
			else:
				pass

			img = img.cuda().float()
			gt_label = gt_label.cuda().long()

			output = feature_extractor(img)
			output = pool(output)
			output_np = output.detach().cpu().numpy()
			target_np = gt_label.detach().cpu().numpy()

			out_output.append(np.squeeze(output_np))
			out_target.append(target_np[:, np.newaxis])

	output_array = np.concatenate(out_output, axis=0)
	target_array = np.concatenate(out_target, axis=0)

	pca, tsne, pca_results, tsne_results  = reduce_then_tsne(output_array, pca=pca, tsne=tsne, fit=fit)
	tsne_vis(tsne_results, target_array, fit)

	return pca, tsne, pca_results, tsne_results

if __name__ == '__main__':

	num_epoch = 1000

	total_data = pd.read_csv('/MULTIX/DATA/HOME/frankenstein_data.csv')

	# mapping = {'rsna': 0, 'sirm': 1, 'cohen':2, 'ricord':3} 
	mapping = {'rsna':0, 'sirm':1, 'ricord':2,'cohen':3}

	total_data['source_map'] = total_data['source'].map(mapping) # map source labels to numeric
	print(total_data['source'].value_counts())

	test_df = total_data[total_data['source_split']=='test']
	train_df = total_data[total_data['source_split']=='train']

	# ========
	
	seed = 0

	target = test_df.source_map
	train_loader, test_loader = make_generators(train_df, test_df, seed)


	for i in range(1,6): # iterate through cv folds

		seg_model = VGGNestedUNet(num_classes=1)

		device = torch.device('cuda')
		seg_model = VGGNestedUNet(num_classes=1)
		seg_model = seg_model.to(device)
		seg_model.eval()
		seg_model.load_state_dict(torch.load(f"/MULTIX/DATA/HOME/vgg_nested_unet_{i}.pth"))

		classifier = DCNN().to(device)
		classifier.eval()
		classifier.load_state_dict(torch.load(f"/home/ubuntu/frankenstein_dcnn_k-{i}.pth"))

		gt_list = []
		pred_list = []
		output_list = []

		# target_names = ['RSNA', 'CHOWDHURY', 'COHEN', 'RICORD']
		target_names = ['RSNA', 'CHOWDHURY', 'RICORD', 'COHEN']

		for v, data in enumerate(test_loader):
			img, gt_label = data

			img = img.cuda().float()

			img = covidx_seg(img)

			gt_label = gt_label.cuda()

			output = classifier(img)
			pred = output.detach().argmax(dim=1, keepdim=True).squeeze()

			# apply grad-cam
			# saliency_img = visualise_cam(img, output, v, classifier)
			for i in range(len(gt_label)):
				gt_list.append(gt_label[i].item())
				pred_list.append(pred[i].item())

				output_list.append(output[i].squeeze().detach().cpu().numpy())


		fpr = dict()
		tpr = dict()
		roc_auc = dict()

		plt.figure()
		for i in range(len(target_names)):
			# multiclass roc curve
			fpr[i], tpr[i], _ = roc_curve(np.array(gt_list),np.array(output_list)[:,i], pos_label=i, sample_weight=None, drop_intermediate=False)
			roc_auc[i] = auc(fpr[i], tpr[i])    
		    
		#    plt.figure()
			lw = 2
			target_n = target_names[i]
			plt.plot(fpr[i], tpr[i],lw=lw, label=f'{target_n} - (area = %0.2f)' % roc_auc[i])
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title(f'Deep CNN')
			plt.legend(loc="lower right")

		plt.savefig(f'/MULTIX/DATA/HOME/covidx_frankenstein_roc_curve.png')

		print(classification_report(gt_list, pred_list, target_names=target_names)) # performance metrics
		print(confusion_matrix(gt_list, pred_list))

		# truncate classifier to final hidden layer
		feature_extractor = nn.Sequential(*(list(classifier.children())[:-1]))
		feature_extractor = classifier.features_conv
		feature_extractor.cuda()
		pool = classifier.avgpool.cuda() # extract pool layer

		# training the pca-tsne algorithms
		pca = PCA(n_components=20)
		tsne = TSNE(n_components=2, verbose = 1)
		pca, tsne, pca_results, tsne_results = pca_tsne(train_loader, pca, tsne, fit=True, segmentation=True)

    	# testing pca-tsna algorithms
		pca, tsne, pca_results, tsne_results = pca_tsne(test_loader, pca, tsne, fit=False, segmentation=True)

		break







