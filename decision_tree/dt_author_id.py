#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn import tree
sys.path.append("../tools/")
from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]


#########################################################
### your code goes here ###


#########################################################
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
print("training time:", round(t1-t0, 3), "s")
# Using the starter code in decision_tree/dt_author_id.py, get a decision tree
# up and running as a classifier, setting min_samples_split=40. It will probably
# take a while to train.
# What’s the accuracy?
accuracy = clf.score(features_test, labels_test)
t2 = time()
print("prediction time:", round(t2-t1, 3), "s")
print('Accuracy:', accuracy)

print("Number of features: {nf}".format(nf=len(features_train[0])))
print(features_train.shape)
