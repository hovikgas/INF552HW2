# INF 552 Homework 2
# Change directory to wherever it is you downloaded the data
# cd C:\Users\hovo_\Google Drive\Classes\PhD\INF 552\Homework\HW2

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn import ensemble

# Import CSV file and get some info on it
df = pd.read_csv('join_data_extract3.csv',header='infer')
df.info()

# convert from pandas to numpy
county = df.values

# split the data to X,y which is attributes and output class (small y)
X,y = county[:,:4], county[:,4]
# This selects all rows then X holds first 6 column attributes and y the last column as class (i.e. 1 dimension array)

#impute missing values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)

# make sure that all the values in numpy are int to avoid potential numpy problems
X,y = X.astype(int), y.astype(int)

# split the data for 70% training and 30% for testing in scikit ML
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

# set up the Random Forest Classifier with 10 trees in the forest
clf = ensemble.RandomForestClassifier(n_estimators=100, verbose=1)

# feed the classifier with the training data
clf.fit(X_train,y_train)

# check the accuracy of the classification on the test data
print(clf.score(X_test,y_test))
# got ~97% <-- not bad!

# see which features are the most important
clf.feature_importances_

'''
# let's see if we can improve it by scaling the data between max and min
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# try it again with the newly scaled data
clf.fit(X_train_scaled,y_train)
clf.score(X_test_scaled,y_test)
# got 98% this time, way better!
'''
# plot the values to see what is going on
y_pred = clf.predict(X_test)

# import matplot so we can plot it
import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.show()

importances = clf.feature_importances_
std = np.std([clf.feature_importances_ for clf in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()