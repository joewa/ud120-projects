#!/usr/bin/python

import os
import sys
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from sklearn.preprocessing     import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.decomposition     import PCA, KernelPCA
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, f_classif, SelectFromModel, RFECV, RFE
from sklearn.model_selection   import train_test_split, StratifiedShuffleSplit, GridSearchCV, ParameterGrid
from sklearn.metrics           import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.tree              import DecisionTreeClassifier
from sklearn.neighbors         import KNeighborsClassifier, NearestCentroid
from sklearn.svm               import LinearSVC, SVC
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import VotingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline          import Pipeline, FeatureUnion
#from sklearn.compose           import ColumnTransformer
# https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer.html#sphx-glr-auto-examples-compose-plot-column-transformer-py

from collections import Counter

# DataFrameSelector from: "Hands on machine learning with Scikit-Learn & Tensorflow"
# Better approach: https://scikit-learn.org/stable/modules/compose.html
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
# was "final_project_dataset.pkl"
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Some utility functions do not like mixed types --
for p_id, p_info in data_dict.items():
    if p_info['email_address'] == 'NaN':
        p_info['email_address'] = 'unknown'
    for info in p_info:
        if p_info[info] == 'NaN':
            p_info[info] = np.nan

data_dict_raw = deepcopy(data_dict)

### Task 2: Remove outliers
# This task has been already done in the Jupyter-Notbook, because the given
# pkl-file was corrupt, i.e. did not represent the full data from the insider
# payments sheet. Please see the notebook for more information.
#with open("final_project_dataset_PROVEN.pkl", "rb") as data_file:
#    data_dict = pickle.load(data_file)


#df = pd.DataFrame.from_dict(data_dict_raw, orient='index').replace('NaN', np.nan)
#toomanynan = df.isna().sum(axis=1) > 16
#df_toomanynan = df[toomanynan]

#outliers2remove = df_toomanynan.index.values.tolist() + [df['salary'].idxmax()]

#for pos in outliers2remove:
#    data_dict.pop(pos, 0) # remove them from the data_dict


data_dict_raw = deepcopy(data_dict)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = deepcopy(data_dict)

# Relevant and engineered features due to "engineering judgement"
# gotten versus deferral payments (sum it up)
df = pd.read_pickle("final_project_dataset_PROVEN.pkl")
df = df.replace(np.nan, 0.0)

f = pd.Series(features_financial_salary)
df['total_payments_corr'] = df[f[~f.isin(['deferred_income'])]].sum(axis=1)

f = pd.Series(features_financial_stock)
df['total_stock_value_corr'] = df[f[~f.isin(['restricted_stock_deferred'])]].sum(axis=1)
df['total_cash'] = df['total_payments_corr'] + df['total_stock_value_corr']

df['total_payments_corr'].replace(0.0, 1.0, inplace=True)
df['total_stock_value_corr'].replace(0.0, 1.0, inplace=True)
df['to_messages'].replace(0.0, 1.0, inplace=True)
df['from_messages'].replace(0.0, 1.0, inplace=True)
df['salary'].replace(0.0, 1.0, inplace=True)

df["bonus_to_salary"]                    = df["bonus"]                     / df["salary"]

df['fraction_salary']                    = df['salary']                    / df['total_payments_corr']
df['fraction_bonus']                     = df['bonus']                     / df['total_payments_corr']
df['fraction_long_term_incentive']       = df['long_term_incentive']       / df['total_payments_corr']
df['fraction_deferral_payments']         = df['deferral_payments']          / df['total_payments_corr']

df['fraction_deferred_income']           = df['deferred_income']           / df['total_payments_corr'] # lost cash

df['fraction_exercised_stock_options']   = df['exercised_stock_options']   / df['total_stock_value_corr']
df['fraction_restricted_stock_deferred'] = df['restricted_stock_deferred'] / df['total_stock_value_corr'] # lost stock
df['fraction_restricted_stock']          = df['restricted_stock']          / df['total_stock_value_corr']

df['fraction_employer_direct_cash']  = df['fraction_salary'] + df['fraction_bonus'] + df['fraction_long_term_incentive']
df['fraction_employer_stock_cash']   = df['total_stock_value'] / df['total_stock_value_corr']
df['fraction_director_direct_cash']  = df['director_fees'] / df['total_payments_corr']

df['fraction_from_poi_to_this_person'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_shared_receipt_with_poi'] = df['shared_receipt_with_poi'] / df['to_messages']
df['fraction_from_this_person_to_poi'] = df['from_this_person_to_poi'] / df['from_messages']


features_fractions_lost = ['fraction_deferred_income', 'fraction_restricted_stock_deferred']
features_fractions_cash = ['fraction_employer_direct_cash', 'fraction_employer_stock_cash']
features_absolute_cash  = ['total_payments_corr', 'total_stock_value_corr', 'bonus_to_salary']
features_email = ['to_messages', 'from_messages', 'fraction_from_poi_to_this_person', 'fraction_shared_receipt_with_poi', 'fraction_from_this_person_to_poi']

data_d = df.to_dict(orient='index') # Required for Udacitiy's test_classifier

my_dataset = data_d
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# A variety of classifiers has been investigated and compared in the Jupyter-Notebook.
# Only the best classifier is included and optimized here.
d_scaler0 = {
        "scaler0": [StandardScaler(), MaxAbsScaler()] # RobustScaler(), MaxAbsScaler(), MinMaxScaler()
    }
d_logreg = {
        "classifier": [LogisticRegression(random_state=42)],
        "classifier__C": [0.02, 0.03, 0.04, 0.05, 0.5, 1, 1e1, 1e2, 1e3, 1e5, 1e10],
        "classifier__tol":[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-10],
        "classifier__class_weight":['balanced'],
        "classifier__solver": ["liblinear"]
    }
d_dt = {
        "classifier": [DecisionTreeClassifier(random_state=42)],
        "classifier__criterion": ["entropy"],
        "classifier__max_depth": [5,6,7,8,9,10,11, None]
        #"classifier__min_samples_leaf": [1,2,3,4,5] # Makes it worse
    }
# These is the the best scoring feature selection(s)
d_bestsel = {
        "selector": [DataFrameSelector()],
        "selector__attribute_names": [
                                  ["total_cash", "fraction_deferred_income", "fraction_restricted_stock_deferred"],
                                  ["total_stock_value", "expenses", "from_messages", "total_stock_value_corr", "total_cash",
                                  "bonus_to_salary", "fraction_exercised_stock_options", "fraction_restricted_stock_deferred",
                                  "fraction_employer_stock_cash", "fraction_from_this_person_to_poi"]
                                 ]
    }
pipe_params = [
    #("selector0", None),
    ("selector", None),
    ("scaler0", None),
    #("dimreducer", None),
    ("classifier", None)
]

pipe = Pipeline(pipe_params)


# Neue Pipeline mit DataFrameSelector am Anfang machen
param_grid_all = [
    #{**d_scaler0, **d_bestsel, **d_PCA_2, **d_dt }, # Accuracy: 0.87207	Precision: 0.52920	Recall: 0.36700	F1: 0.43342	F2: 0.39097
    {**d_scaler0, **d_bestsel, **d_logreg}, # Accuracy: 0.77980	Precision: 0.36862	Recall: 0.91400	F1: 0.52536	F2: 0.70530


    #{**d_scaler0, **d_kNearestCentroid}, # Most simple one!
    #{**d_scaler0, **d_bestsel, **d_kNearestCentroid}, # F1: WORKS

    #{**d_scaler0, **d_bestsel, **d_rforest} # Remove to make it work!
]

sss = StratifiedShuffleSplit(n_splits=100, random_state=42) # Applied for ALL models
clf_best_results = []

for param_grid_clf in param_grid_all:
    pg=[param_grid_clf]
    print(pg)
    grid_search = GridSearchCV(pipe, param_grid=pg, cv=sss, scoring="f1", n_jobs=1)

    f = pd.Series(df.columns)
    X_cols = f[~f.isin(['poi'])]
    grid_search.fit(df[X_cols], df['poi'].values)
    clf_best_results.append( {
            "score": grid_search.best_score_,
            "best_estimator": grid_search.best_estimator_,
            "best_params_": grid_search.best_params_
        } )
# Prepare the final pipeline that will work with test_classifier
df_best_results = pd.DataFrame.from_dict(clf_best_results)
best_estimator = df_best_results.loc[ df_best_results["score"].idxmax(), "best_estimator" ]
print("Best F1 score:{}".format(df_best_results["score"].max()))

pipe_final_params = [
    ("scaler0", best_estimator.named_steps["scaler0"]),
    #("dimreducer", best_estimator.named_steps["dimreducer"]),
    ("classifier", best_estimator.named_steps["classifier"])
]

pipe_final = Pipeline(pipe_final_params)
features_final = best_estimator.named_steps["selector"].attribute_names


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

dump_classifier_and_data(pipe_final, my_dataset, features_final)
