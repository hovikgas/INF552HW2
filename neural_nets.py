# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 23:08:21 2016

Predicting number of child injuries 
in the age category of less than 1 year old 
using census tract data regarding children age/sex data. 

@author: Willa Cheng
"""
# Import data
import pandas as pd
import numpy as np
from sklearn import preprocessing
data = pd.read_csv('homework2_data.csv',index_col = 0)

y_1 = data.loc[:,["Sum of Less than 1"]]
y_2 = data.loc[:,["Sum of One Year Old"]]
y_3 = data.loc[:,["Sum of Two Years Old"]]

# Transforming response variables
y1_org = y_1.as_matrix()
y2 = y_2.as_matrix()
y3 = y_3.as_matrix()
# Standardize
y1 = preprocessing.scale(y1_org)

# Transforming input variables
X_org = data.iloc[:,4:19]
X_org = X_org.as_matrix()
# Standardize
X = preprocessing.scale(X_org)

# Apply Neural Network

from NN import *
NN = Neural_Network()
T = trainer(NN)

# Cross Validation with K-Fold
num_folds = 7
subset_size = len(y1)/num_folds

for i in range(num_folds):
    X_testing_this_round = X[i*subset_size:,:][:subset_size,:]
    X_training_this_round = np.append(X[:i*subset_size,:] , X[(i+1)*subset_size:,:],axis = 0)
    y1_testing_this_round = y1[i*subset_size:][:subset_size]
    y1_training_this_round = np.append(y1[:i*subset_size] , y1[(i+1)*subset_size:],axis = 0)
    # Train the model
    T.train(X_training_this_round, y1_training_this_round)
    NN.costFunctionPrime(X_training_this_round,y1_training_this_round)
    # predicting value
    yHat = NN.forward(X_testing_this_round)
    y1_testing_this_round = y1_testing_this_round.ravel()
    yHat = yHat.ravel()
    # Evaluate with Mean Squared Error
    MSE = sum((y1_testing_this_round - yHat)**2)/subset_size
    print(MSE)


