import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]

def process_image_file(filepath, img_size):
    img = cv2.imread(filepath)
    img = cv2.resize(img, img_size)
    return img

def random_ratio_resize(img, prob=0.3, delta=0.1):
    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 480 or size[1] > 480:
        print(img.shape, size, ratio)

    img = cv2.resize(img, size)
    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT,
                             (0, 0, 0))

    if img.shape[0] != 480 or img.shape[1] != 480:
        raise ValueError(img.shape, size)
    return img


_augmentation_transform = ImageDataGenerator(
    rescale = 1/255.0,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=(0.85, 1.15),
    fill_mode='constant',
    cval=0.,)

def apply_augmentation(img):
    img = random_ratio_resize(img)
    img = _augmentation_transform.random_transform(img)
    return img

def _process_csv_file(file):
    mapping={
                'non-pneumonia': 0,
                'pneumonia': 1,
                'COVID-19': 2
            }
    data = pd.read_csv(file)
    print(data.finding)
    data['three_class'] = data['finding'].replace(mapping)
    return data


class BalanceCovidDataset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir,
            csv_file,
            is_training=True,
            batch_size=8,
            input_shape=(480, 480),
            n_classes=3,
            num_channels=3,
            mapping={
                'non-pneumonia': 0,
                'pneumonia': 1,
                'COVID-19': 2
            },
            shuffle=False,
            augmentation=apply_augmentation,
            covid_percent=0.3,
            pneum_percent=0.3,
            class_weights=[1., 1., 1.],
            top_percent=0.08
    ):
        'Initialization'
        self.datadir = data_dir
        self.dataset = _process_csv_file(csv_file)
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.covid_percent = covid_percent
        self.pneum_percent = pneum_percent
        self.class_weights = class_weights
        self.n = 0
        # self.augmentation = augmentation
        self.top_percent = top_percent

        if self.is_training:
            self.dataset = self.dataset[self.dataset['split']=='train']
        else:
            self.dataset = self.dataset[self.dataset['split']=='val']

        self.dataset = self.dataset.reset_index(drop=True)

        # split classes into separate dataframes
        negative_df = self.dataset[self.dataset['finding']=='non-pneumonia']
        negative_df = negative_df.reset_index(drop=True)
        
        pneumonia_df = self.dataset[self.dataset['finding']=='pneumonia']
        pneumonia_df = pneumonia_df.reset_index(drop=True)
        
        COVID_df = self.dataset[self.dataset['finding']=='COVID-19']
        COVID_df = COVID_df.reset_index(drop=True)

        datasets = {'negative': negative_df, 'pneumonia': pneumonia_df, 'COVID-19': COVID_df}

        self.datasets = [
            datasets['negative'], datasets['pneumonia'],
            datasets['COVID-19'],
        ]

        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y, weights = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return batch_x, batch_y, weights

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v.values)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros(
            (self.batch_size, *self.input_shape,
             self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0].iloc[idx * self.batch_size:(idx + 1) *
                                       self.batch_size, :]

        covid_size = max(int(len(batch_files) * self.covid_percent), 1)

        covid_inds = np.random.choice(np.arange(len(batch_files)),
                                      size=covid_size,
                                      replace=False)

        covid_files = self.datasets[2].iloc[np.random.choice(self.datasets[2].index, covid_size, replace=False)]

        for i in range(covid_size): # collect covid-19
            batch_files.iloc[covid_inds[i]] = covid_files.iloc[i]

        pneum_size = max(int(len(batch_files) * self.pneum_percent), 1)

        pneum_inds = np.random.choice(np.arange(len(batch_files)),
                                      size=pneum_size,
                                      replace=False)

        pneum_files = self.datasets[1].iloc[np.random.choice(self.datasets[1].index, covid_size, replace=False)]

        for i in range(pneum_size): # collect pneumonia
            batch_files.iloc[pneum_inds[i]] = pneum_files.iloc[i]

        for i in range(len(batch_files)): # collect full batch
            sample = batch_files.iloc[i]

            x = process_image_file(sample.filename, self.input_shape)

            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation(x)

            x = x.astype('float32') / 255.0
            y = sample.three_class

            batch_x[i] = x
            batch_y[i] = y

        class_weights = self.class_weights
        weights = np.take(class_weights, batch_y.astype('int64'))

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes), weights
