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

from covidx_frankenstein import DCNN, CustomTensorDataset


def make_generators(train_df, test_df, seed):
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    transform = A.Compose(
        [
          A.Resize(224,224),
          ToTensorV2()
        ]
    )

    train_data = CustomTensorDataset(train_df, transform=transform)
    train_loader = DataLoader(train_data, batch_size=24, num_workers=0,
                              pin_memory=True, shuffle=True, 
                              worker_init_fn=np.random.seed(seed))


    test_data =CustomTensorDataset(test_df, transform=transform)
    test_loader = DataLoader(test_data, batch_size=24, num_workers=0,
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

        heatmap = cv2.resize(heatmap, (224,224))

        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = img.squeeze().permute(1,2,0)
        # img = img.squeeze().unsqueeze(0).permute(1,2,0)

        superimposed_img = heatmap * 0.3 + (img*255).cpu().numpy()
        cv2.imwrite('/app/frankenstein_covidx_grad_cam_{}_{}.png'.format(pred_ind.cpu().detach().numpy(), v), superimposed_img.squeeze())
        
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
	    print(np.unique(indices))
	    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1],label=target_names[cl], alpha=0.5)
	    plt.xlabel("Dimension 1")
	    plt.ylabel("Dimension 2")

	if fit == True:
		title = "/app/tsne_plot_train.png"
	else:
		title = "/app/tsne_plot_test.png"
	plt.legend()
	plt.savefig(title)
	# plt.show()


def pca_tsne(data_loader, pca, tsne, fit=True):
	out_target = []
	out_output =[]

	for data in tqdm(data_loader):
		img, gt_label = data
		img = img.cuda().float()
		gt_label = gt_label.cuda().long()

		output = feature_extractor(img)
		output = pool(output)
		output_np = output.detach().cpu().numpy()
		target_np = gt_label.detach().cpu().numpy()

		out_output.append(np.squeeze(output_np))
		out_target.append(target_np[:, np.newaxis])

	output_array = np.concatenate(out_output, axis=0)
	print(output_array.shape)
	target_array = np.concatenate(out_target, axis=0)
	print(target_array.shape)

	pca, tsne, pca_results, tsne_results  = reduce_then_tsne(output_array, pca=pca, tsne=tsne, fit=fit)
	tsne_vis(tsne_results, target_array, fit)

	return pca, tsne, pca_results, tsne_results

if __name__ == '__main__':

	num_epoch = 1000

	total_data = pd.read_csv('/app/full_covidx_data2.csv')

	mapping = {'rsna': 0, 'sirm': 1, 'cohen':2, 'ricord':3} 

	total_data['source_map'] = total_data['source'].map(mapping) # map source labels to numeric
	print(total_data['source'].value_counts())

	test_df = total_data[total_data['source_split']=='test']
	train_df = total_data[total_data['source_split']=='train']

	# ========
	
	num_epoch = 1000
	
	seed = 0

	target = test_df.source_map
	train_loader, test_loader = make_generators(train_df, test_df, seed)


	for i in range(1,4): # iterate through cv folds

		classifier = DCNN().to('cuda')
		classifier.eval()
		classifier.load_state_dict(torch.load(f"/app/source_classifier_k-{i}.pth"))

		gt_list = []
		pred_list = []
		output_list = []

		target_names = ['RSNA', 'CHOWDHURY', 'COHEN', 'RICORD']
		
		for v, data in enumerate(test_loader):
			img, gt_label = data

			img = img.cuda().float()
			gt_label = gt_label.cuda()

			output = classifier(img)
			pred = output.detach().argmax(dim=1, keepdim=True).squeeze()

			# apply grad-cam
			# saliency_img = visualise_cam(img, output, v, classifier)

			gt_list.append(gt_label.item())
			pred_list.append(pred.item())
			output_list.append(output.squeeze().detach().cpu().numpy())


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

		plt.savefig(f'/app/covidx_frankenstein_roc_curve.png')

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
		pca, tsne, pca_results, tsne_results = pca_tsne(train_loader, pca, tsne)

    	# testing pca-tsna algorithms
		pca, tsne, pca_results, tsne_results = pca_tsne(test_loader, pca, tsne, fit=False)

		break






