#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn import svm

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

# For testing: Reduce training set to speed up...
# One way to speed up an algorithm is to train it on a smaller training dataset.
# The tradeoff is that the accuracy almost always goes down when you do this.
# Let’s explore this more concretely: add in the following two lines immediately
# before training your classifier.
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
# These lines effectively slice the training dataset down to 1% of its original
# size, tossing out 99% of the training data. You can leave all other code unchanged.
# What’s the accuracy now?

# Train a SVM , kernel='linear'
clf = svm.SVC(C=1.0, kernel='linear', gamma='auto')
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
t1 = time()
accuracy = clf.score(features_test, labels_test)
print("prediction time:", round(time()-t1, 3), "s")
print('Accuracy:', accuracy)


# Train a SVM , kernel='linear'
# Keep the training set slice code from the last quiz, so that you are still
# training on only 1% of the full training set. Change the kernel of your SVM
# to “rbf”. What’s the accuracy now, with this more complex kernel?
clf = svm.SVC(C=10000.0, kernel='rbf', gamma='auto')
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
print("training time:", round(t1-t0, 3), "s")

accuracy = clf.score(features_test, labels_test)
t2 = time()
print("prediction time:", round(t2-t1, 3), "s")
print('Accuracy:', accuracy)


# Keep the training set size and rbf kernel from the last quiz, but try several
# values of C (say, 10.0, 100., 1000., and 10000.). Which one gives the best
# accuracy?
for c in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
    print("C=", c)
    clf = svm.SVC(C=c, kernel='rbf', gamma='auto')
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3), "s")
    t1 = time()
    accuracy = clf.score(features_test, labels_test)
    print("prediction time:", round(time()-t1, 3), "s")
    print('Accuracy:', accuracy)
# Note: A grater value of C corrosponds to a more complex descision boundary.

print("Some predicions:")
for id in [10, 26, 50]:
    print("Mail id:", id, "prediction:", clf.predict([features_test[id]]) )


# Question: There are over 1700 test events--how many are predicted to be in
# the “Chris” (1) class? (Use the RBF kernel, C=10000., and the full training
# set.)
clf = svm.SVC(C=10000.0, kernel='rbf', gamma='auto')
clf.fit(features_train, labels_train)
print("There are over 1700 test events--how many are predicted to be in the “Chris” (1) class? :")
all_predicions = clf.predict(features_test)
from collections import Counter
print(Counter(all_predicions))

# Question:
# Now that you’ve optimized C for the RBF kernel, go back to using the full
# training set. In general, having a larger training set will improve the
# performance of your algorithm, so (by tuning C and training on a large dataset)
# we should get a fairly optimized result. What is the accuracy of the optimized SVM?
# --> 0.99
