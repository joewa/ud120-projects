#!/usr/bin/python

import os
import sys
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from sklearn.preprocessing     import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition     import PCA, KernelPCA
from sklearn.feature_selection import SelectPercentile, SelectKBest, SelectFromModel, RFECV, RFE
from sklearn.model_selection   import train_test_split, StratifiedShuffleSplit, GridSearchCV, ParameterGrid
from sklearn.metrics           import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.tree              import DecisionTreeClassifier
from sklearn.neighbors         import KNeighborsClassifier, NearestCentroid
from sklearn.svm               import LinearSVC, SVC
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline          import Pipeline, FeatureUnion


from collections import Counter

# DataFrameSelector from: "Hands on machine learning with Scikit-Learn & Tensorflow"
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names=['total_payments', 'total_stock_value']):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


sys.path.append("../tools/")
sys.path.append("../choose_your_own/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_email_stats = ['from_messages','from_this_person_to_poi', 'to_messages','from_poi_to_this_person', 'shared_receipt_with_poi']
features_financial_salary = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees']
features_financial_salary_total = ['total_payments']
features_financial_stock = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred']
features_financial_stock_total = ['total_stock_value']
features_email = ['fraction_from_poi_to_this_person', 'fraction_from_this_person_to_poi'] # This is designed in Task 3

features_list = ['poi'] + ['salary', 'bonus'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Some utility functions do not like mixed types
for p_id, p_info in data_dict.items():
    # p_info['poi'] = float(int(p_info['poi']))
    #if p_info['poi']:
    #    p_info['poi'] = True
    #else:
    #    p_info['poi'] = False
    if p_info['email_address'] == 'NaN':
        p_info['email_address'] = 'unknown'
    for info in p_info:
        if p_info[info] == 'NaN':
            p_info[info] = np.nan

data_dict_raw = deepcopy(data_dict)

### Task 2: Remove outliers
df = pd.DataFrame.from_dict(data_dict_raw, orient='index').replace('NaN', np.nan)
toomanynan = df.isna().sum(axis=1) > 16
df_toomanynan = df[toomanynan]

outliers2remove = df_toomanynan.index.values.tolist() + [df['salary'].idxmax()]

for pos in outliers2remove:
    data_dict.pop(pos, 0) # remove them from the data_dict



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = deepcopy(data_dict)

# Let the fraction of emails from and to a person of interest be new (human readable) features.
for p_id, p_info in my_dataset.items():
    p_info['fraction_from_poi_to_this_person'] = p_info['from_poi_to_this_person'] / p_info['to_messages']
    p_info['fraction_from_this_person_to_poi'] = p_info['from_this_person_to_poi'] / p_info['from_messages']
    for info in p_info:
        if p_info[info] != p_info[info]: # NaN is always != NaN
            p_info[info] = 0.0


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection  import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
