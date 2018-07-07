# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:10:19 2018

@author: Peter Liu
"""

import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10

#load data
dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = load_CIFAR10(dir)
print('training data shape', X_train.shape)
print('training label shape', Y_train.shape)
print('test data shape', X_test.shape)
print('test label shape', Y_test.shape)

Classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
sample_per_class =  7
num_classes = len(Classes)
for y, cls in enumerate(Classes):
    idxs = np.flatnonzero(Y_train==y)
    idxs = np.random.choice(idxs, sample_per_class,replace=False)
    for i, idx in enumerate(idxs):
        plt_idxs=i*num_classes+y+1
        plt.subplot(sample_per_class,num_classes,plt_idxs)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

mask = range(num_training,num_training+num_validation)
X_val = X_train[mask]
Y_val = Y_train[mask]

mask = range(num_training)
X_train = X_train[mask]
Y_train = Y_train[mask]

mask = np.random.choice(num_training,num_dev,replace=False)
X_dev = X_train[mask]
Y_dev = Y_train[mask]

mask = range(num_test)
X_test = X_test[num_test]
Y_test = Y_test[num_test]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', Y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', Y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)





