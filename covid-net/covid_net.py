from __future__ import print_function

import tensorflow as tf
import os, argparse, pathlib

import pandas as pd
import matplotlib.pyplot as plt

from covid_net_loader import BalanceCovidDataset
from covid_net_eval import eval

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--weightspath', default='/root/my-repo/covid-net/COVIDNet-CXR3-C/', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-18540', type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='/root/my-repo/open-data-study/open-data-files/balanced_covidx_data.csv', type=str, help='Path to train file')
parser.add_argument('--testfile', default='/root/my-repo/open-data-study/open-data-files/balanced_covidx_data.csv', type=str, help='Path to test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/root/my-repo/data/', type=str, help='Path to data folder')
parser.add_argument('--covid_weight', default=1., type=float, help='Class weighting for covid')
parser.add_argument('--pneum_weight', default=1., type=float, help='Class weighting for pneumonia')
parser.add_argument('--neg_weight', default=1., type=float, help='Class weighting for pneumonia')


parser.add_argument('--covid_percent', default=0.3, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.00, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_1/MatMul:0', type=str, help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str, help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str, help='Name of sample weights tensor for loss')


args = parser.parse_args()

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 10

# output path
outputPath = '/root/my-repo/results/'
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

mapping={'non-pneumonia': 0, 'pneumonia': 1, 'COVID-19': 2}
data = pd.read_csv(args.trainfile)
data['three_class'] = data['finding'].replace(mapping)
train_df = data[data['split']=='train']

def make_generators(train_df, val_df):

    generator = BalanceCovidDataset(data_dir=args.datadir,
                                    csv_file=train_df,
                                    batch_size=args.bs,
                                    input_shape=(args.input_size, args.input_size),
                                    covid_percent=args.covid_percent,
                                    class_weights=[args.neg_weight, args.pneum_weight, args.covid_weight],
                                    top_percent=args.top_percent)

    val_generator = BalanceCovidDataset(data_dir=args.datadir,
                                    csv_file=val_df,
                                    batch_size=args.bs,
                                    input_shape=(args.input_size, args.input_size),
                                    covid_percent=args.covid_percent,
                                    class_weights=[args.neg_weight, args.pneum_weight, args.covid_weight],
                                    top_percent=args.top_percent,
                                    is_training=False)
    return generator, val_generator

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    labels_tensor = graph.get_tensor_by_name(args.label_tensorname)
    sample_weights = graph.get_tensor_by_name(args.weights_tensorname)
    pred_tensor = graph.get_tensor_by_name(args.logit_tensorname)
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # save base model
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    print('Baseline eval:')

    test_df = data[data['split']=='test']

    eval(sess, graph, test_df, args.datadir,
         args.in_tensorname, args.out_tensorname, args.input_size)

    # Training cycle
    print('Training started')
    total_batch = len(generator)
    progbar = tf.keras.utils.Progbar(total_batch)

    best_loss = 1e10
    no_improvement = 0
    patience = 50

    seed = 0
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    target = train_df.three_class

    training_times = []

    fold_no = 1
    for train_idx, val_idx in kfold.split(train_df, target):
        train_kfold = train_df.iloc[train_idx]
        val_kfold = train_df.iloc[val_idx]

        generator, val_generator = make_generators(train_kfold, val_kfold)

        train_loss_list = []
        val_loss_list = []

        best_loss = 1e10
        no_improvement = 0
        patience = 50

        for epoch in range(args.epochs):
            for i in range(total_batch):
                # Run optimization
                batch_x, batch_y, weights = next(generator)
                sess.run(train_op, feed_dict={image_tensor: batch_x,
                                              labels_tensor: batch_y,
                                              sample_weights: weights})
                progbar.update(i+1)

            if epoch % display_step == 0:
                # Monitor train loss changes
                pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
                loss = sess.run(loss_op, feed_dict={pred_tensor: pred,
                                                    labels_tensor: batch_y,
                                                    sample_weights: weights})

                print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
                
                # Monitor val loss changes
                val_batch_x, val_batch_y, weights = next(val_generator)
                val_pred = sess.run(pred_tensor, feed_dict={image_tensor:val_batch_x})
                val_loss = sess.run(loss_op, feed_dict={pred_tensor: val_pred,
                                                    labels_tensor: val_batch_y,
                                                    sample_weights: weights})

                print("Epoch:", '%04d' % (epoch + 1), "Minibatch val loss=", "{:.9f}".format(val_loss))

                train_loss_list.append(loss)
                val_loss_list.append(val_loss)
                
                # Evalutate predictions on val set
                eval(sess, graph, val_kfold, args.datadir,
                     args.in_tensorname, args.out_tensorname, args.input_size)

                if val_loss < best_loss:
                    best_loss = val_loss
                    saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=False)
                    print('Saving checkpoint at epoch {}'.format(epoch + 1))
                    no_improvement = 0
                    print("no improvements in validation loss - {}/{}".format(no_improvement, patience))

                elif val_loss > best_loss:
                    no_improvement = no_improvement + 1
                    print("no improvements in validation loss - {}/{}".format(no_improvement, patience))

                    if no_improvement == patience:
                        print("early stopped")
                        break

    plt.figure()
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.legend()
    plt.title('losses')
    plt.savefig('/home/ubuntu/covid-net-loss-1500.png')








