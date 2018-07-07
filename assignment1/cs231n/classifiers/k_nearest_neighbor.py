# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 15:49:18 2018

@author: Peter Liu
"""
import numpy as np
class KNearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self,X,y):
        self.X_train = X
        self.Y_train = y
        
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists=self.compute_distances_no_loops(X)
            
        elif num_loops == 1:
            dists=self.compute_distances_one_loop(X)
            
        elif num_loops == 2:
            dists=self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalide value %d for num_loops' % num_loops)
        
        return self.predict_labels(dists,k=k)
    
    def compute_distances_two_loops(self,X):
        num_test=X.shape[0]
        num_train=self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                a = X[i]-self.X_train[j]
                dists[i][j]=np.sum(a**2)
                
        return dists
    
    def compute_distances_one_loop(self,X):
         num_test = X.shape[0]
         num_train = self.X_train.shape[0]
         dists = np.zeros((num_test, num_train))
         #Y=np.zeros((num_test,32*32*3))
         for i in range(num_test):
             #Y[i,:] = X[i].reshape(-1)
             #X_train = self.X_train.reshape(num_train,-1)
             #D = np.tile(Y[i],(num_train,1))-X_train
             #Q = np.sqrt(np.sum(D**2,axis=1))
             dists[i,:]=np.sum((self.X_train-X[i,:])**2,axis=1)
             #if i == 0:
              #print(D,D.shape)
              #print(Q,Q.shape)
              #print(dists)
         return dists
    
    def compute_distances_no_loops(self,X):
        num_test=X.shape[0]
        num_train=self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = -2*np.dot(X, self.X_train.T) + np.sum(X**2, axis = 1)[:, np.newaxis] + np.sum(self.X_train**2, axis = 1)

    #########################################################################
        #dists = np.sum((X[:,np.newaxis,:]-self.X_train)**2,axis=2)
        #Y = np.repeat(X,num_train,axis=0)
        #Z = np.sum((Y-self.X_train)**2,axis=1)
       # dists = Z.reshape((num_test,num_train))
        
                
        return dists
    
    def predict_labels(self,dists,k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = np.argsort(dists[i,:])[0:k]
            tu=sorted([(np.sum(closest_y==j),j) for j in set(closest_y)])
            y_pred[i]=tu[-1][0]
            #y_pred(i)=0
        return y_pred
        
        
                        
            