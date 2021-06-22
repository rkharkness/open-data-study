from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

import pandas as pd
from covid_net_loader import process_image_file


from os.path import isfile, join


def convert_structured_data(data_dir):
    '''Function to convert structured dataset into pandas DataFrame - to make compatible with exisitng covid-net dataloader'''

    full_files = []
    for root, dirs, files in os.walk(os.path.abspath(data_dir)):
        for file in files:
            full_files.append(os.path.join(root, file))

    finding = [i.split('/')[5] for i in full_files] # may need to change index depending on number of dir in full path

    struc_dict = {'path':full_files, 'finding':finding}
    test_df = pd.DataFrame(struc_dict)
    print(test_df.head())

    return test_df

def test_process_csv_file(test_df, data_dir, test_name):
    if test_name == 'CUSTOM':
        test_df = convert_structured_data(data_dir)

    elif test_name == 'COVIDx':
        test_df = test_df[test_df['split']=='test']
        test_df = test_df.reset_index(drop=True)

    else;
        print("Incorrect test name input - must be CUSTOM or COVIDx")

     mapping={
                 'non-pneumonia': 0,
                 'Negative':0,
                 'Pneumonia': 1,
                 'COVID-19': 2
             }
#     data = data[data['split']=='test']
     test_df['three_class'] = test_df['finding'].replace(mapping)
     return test_df

def eval(sess, graph, testfile, testfolder, input_tensor, output_tensor, input_size, save_dir, data_dir, test_name):
    image_tensor = graph.get_tensor_by_name(input_tensor)
    pred_tensor = graph.get_tensor_by_name(output_tensor)
    testfile = test_process_csv_file(test_df, data_dir, test_name)

    y_test = []
    pred_val = []
    pred = []
    for col, row in testfile.iterrows():
        x = process_image_file(row['path'])
        x = x.astype('float32') / 255.0
        y_test.append(row.three_class)
        
        pr = np.array(sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)}))
        
        pred_val.append(pr)
        pred.append(pr.argmax(axis=1))

    y_test = np.array(y_test)
    pred = np.array(pred)
    pred_val = np.array(pred_val)
    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)
    #class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))
    print(classification_report(y_test, pred))

    if test_name == 'COVIDx'
        target_names = ['No_findings', 'Pneumonia', 'Covid-19'] # for COVIDx data
    elif test_name == 'CUSTOM'
        target_names = ['Pneumonia-negative', 'Pneumonia', 'COVID-19'] # for external and LTHT data
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    pred_val = np.squeeze(pred_val)

    plt.figure()
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test, pred_val[:,i], pos_label=i, 
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
        plt.title(f'Receiver operating curve')
        plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(save_dir, f'covid-net_{test_name}_roc_curve_{k}.png'))
 
# +
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
     parser.add_argument('--test_name', type=str, help='Either COVIDx or CUSTOM')
     parser.add_argument('--weightspath', default='models/COVIDNet-CXR4-A', type=str, help='Path to output folder')
     parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
     parser.add_argument('--savedir', default='~/', type=str, help='Name of dir to save results to')
     parser.add_argument('--ckptname', default='checkpoint', type=str, help='Name of model ckpts')
     parser.add_argument('--latest_checkpoint', default='covid-net/saver_covid-net_model_k')
     parser.add_argument('--testfile', default='/app/custom_data_large_chexpert.csv', type=str, help='Name of testfile')
     parser.add_argument('--testfolder', default='data/test', type=str, help='Folder where test data is located')
     parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
     parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
     parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
     parser.add_argument('--data_dir', default= "~/structured_data", type=str, help='Full path to top dir of structured custom dataset')
     args = parser.parse_args()

     import tensorflow.compat.v1 as tf
     tf.disable_v2_behavior()

     # loading in trained covid-net
     k=1 # change for each model version from training under cross validation
     sess = tf.Session()
     tf.get_default_graph()
     saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
     saver.restore(sess, tf.train.latest_checkpoint(args.latest_checkpoint + '_' + str(k)))
             
     graph = tf.get_default_graph()

     # covid-net evaluation
     eval(sess, graph, args.testfile, args.testfolder, args.in_tensorname, args.out_tensorname, args.input_size, args.save_dir,
        args.data_dir, args.test_name)



