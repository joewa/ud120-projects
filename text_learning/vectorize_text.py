#!/usr/bin/python

import os
import pickle
import re
import sys
import numpy as np

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        #if temp_counter < 200:
        if temp_counter > 0: # Always True
            path = os.path.join('..', path[:-1])
            #print(path)
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            email_stemmed = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            words2replace = ["sara", "shackleton", "chris", "germani", "germany", "sshackle", "cgerman", "fax"]
            for word2replace in words2replace:
                email_stemmed = email_stemmed.replace(word2replace, "")

            ### append the text to word_data
            word_data.append(email_stemmed)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append(0)
            elif name == "chris":
                from_data.append(1)
            else:
                from_data.append(np.NaN)


            email.close()


print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )

word_data[10]
print(word_data[152])


### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(word_data)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)

vector = vectorizer.get_feature_names()



#vect = TfidfVectorizer(stop_words="english")
#vect.fit_transform(word_data)
#words_feature = vect.get_feature_names()
print("number_of_unique_words:{}".format(len(vector)))
#print(vect.get_feature_names()[34597])
vector[12346:12390]
