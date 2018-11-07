#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn.tree    import DecisionTreeClassifier
from sklearn.model_selection  import  train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)


### your code goes here
overfit_clf = DecisionTreeClassifier()
overfit_clf = overfit_clf.fit(features, labels)
overfit_pred = overfit_clf.predict(features)
overfit_acc = accuracy_score(labels, overfit_pred)
print(overfit_acc)


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print(confusion_matrix(labels_test, pred))
print(len(labels_test))
print("Accuracy score of test set {}".format(accuracy_score(labels_test, pred)))

# Number of POIs in Test Set
from collections import Counter
print("The Number of POIs in Test Set is:", Counter(pred)[1.0])

# Number of People in Test Set
print("The Number of People in Test Set is:", len(pred))

poi_predicted = [location for location in range(len(pred)) if pred[location] == 1.0]
poi_true      = [location for location in range(len(labels_test)) if labels_test[location] == 1.0]
print ("How many true positives are in the comparison between model prediction & true test labels?")
print (accuracy_score(poi_true, poi_predicted))
poi_true


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
cm = confusion_matrix(true_labels, predictions)
print(cm, '\n')
print(precision_score(true_labels, predictions))
print(recall_score(true_labels, predictions))
6/8
