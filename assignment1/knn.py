# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 11:57:46 2018

@author: Peter Liu
"""

import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10 

cifar10_dir = 'cs231n\datasets\cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
print('training data shape', X_train.shape)
print('training data type', type(X_train))
print('training label shape', Y_train.shape)

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

num_sample = 500
mask1 = list(range(num_sample))
X_train = X_train[mask1]
Y_train = Y_train[mask1]

num_test =50
mask2 = list(range(num_test))
X_test = X_test[mask2]
Y_test = Y_test[mask2]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train, Y_train) 

dists = classifier.compute_distances_one_loop(X_test)
print(dists.shape)

plt.imshow(dists, interpolation='none')
plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
Y_test_pred = classifier.predict_labels(dists,k=1)
# Compute and print the fraction of correctly predicted examples
correct=np.sum(Y_test_pred==Y_test)
accuracy = float(correct)/num_test
print('The prediction accuracy:%f' %accuracy)

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
    
# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')


# Let's compare how fast the implementations are
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# you should see significantly faster performance with the fully vectorized implementation

#X_train = X_train.reshape(50,-1)
#d = np.zeros((5,X_train.shape[1]))
#for i in range(5):
#    d[i,:]=X_train[i,:]

##Cross validation
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds=np.array_split(X_train,num_folds)
y_train_folds=np.array_split(Y_train,num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}
################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for kvalue in k_choices:
    k_to_accuracies[kvalue]=[]
    classifier = KNearestNeighbor()
    for j in range(num_folds):
        Xcsv_train = np.concatenate( [X_train_folds[i] for i in range(num_folds) if i != j]) 
        Ycsv_train = np.concatenate( [y_train_folds[i] for i in range(num_folds) if i != j]) 
        Xcsv_valid = X_train_folds[j]
        Ycsv_valid = y_train_folds[j]
        classifier.train(Xcsv_train,Ycsv_train)
        numcsv_valid= Xcsv_valid.shape[0]
        dists = classifier.compute_distances_one_loop(Xcsv_valid)
        Y_test_pred = classifier.predict_labels(dists,k=kvalue)
        correct=np.sum(Y_test_pred-Ycsv_valid==0) 
        k_to_accuracies[kvalue].append(correct/numcsv_valid)

#print(k_to_accuracies)
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
#for j in range         
#rint('the kvalue is %d', k_to_accuracies[j] 'the accuracy is %d'  )

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
#for k in sorted(k_to_accuracies):
 #   for accuracy in k_to_accuracies[k]:
 #       print('k = %d, accuracy = %f' % (k, accuracy))
 # plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 1

classifier = KNearestNeighbor()
classifier.train(X_train, Y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
