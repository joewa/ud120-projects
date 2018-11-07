#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
print("25. Quiz: {}".format(features_train.shape))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)


print("26. Quiz: ########### stats on test dataset ###########")
print("r-squared score:{0}".format(clf.score(features_test, labels_test)))

top_features = [(number, feature, vectorizer.get_feature_names()[number]) \
                for number, feature in zip(range(len(clf.feature_importances_)), clf.feature_importances_) \
                if feature > 0.2]
max_importance = clf.feature_importances_.max()
max_pos = clf.feature_importances_.argmax()
max_name = vectorizer.get_feature_names()[max_pos]
print("27. Quiz: Importance {0}; Number {1}; Name {2}".format(max_importance, max_pos, max_name))

print("27. Quiz {}".format(top_features))


print("########### stats on training dataset ###########")
print("r-squared score:{0}".format(clf.score(features_train, labels_train)))
