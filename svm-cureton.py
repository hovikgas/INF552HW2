# -*- coding: utf-8 -*-
# INF 552 Homework Number 2 Team 2 (The Habit): Ken Cureton

# Invoke pandas library as pd and numpy as np
import pandas as pd
import numpy as np

# Import CSV file and get info on it
df = pd.read_csv('hw2b.csv',header=0,sep=',')
df.info()

# convert from pandas to numpy
mssa = df.values

# split the data to X,y which is attributes (X) and output class (y)

X,y = mssa[:,7:10], mssa[:,6]

# This selects all rows then X holds feature attributes of interest and y as class (i.e. 1 dimension array)

# make sure that all the values in numpy are int to avoid potential numpy problems
X,y = X.astype(float), y.astype(int)
print ('X', X)
print ('y', y)

# import scikit learn
import sklearn

# import the cross validation module from scikit learn
from sklearn import cross_validation

# split the data for 80% training and 20% for testing in scikit ML
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X,y,test_size=0.5,random_state=0)

# import ensemble from scikit
from sklearn import ensemble

# set up the Random Forest Classifier with 100 trees
clf = ensemble.RandomForestClassifier(n_estimators=100, verbose=1)

# fit the classifier with the training data
clf.fit(X_train,y_train)

# check the accuracy of the classification on the test data
print ('accuracy', clf.score(X_test,y_test))

# see which how many features were used and which are the most important
print ('features', clf.n_features_)
print ('importances', clf.feature_importances_)

# plot the values to see what is going on
y_pred = clf.predict(X_test)

# import matplot so we can plot it
import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.show()

# import SVC from scikit
from sklearn.svm import SVC

#use SVC
clf = SVC(kernel='linear', C=500)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
print (clf.score(X_test, y_test))
print (y_pred)
print (clf.get_params())
print (clf.decision_function(X_test))

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
