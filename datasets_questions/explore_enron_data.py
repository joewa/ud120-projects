#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
print(len(enron_data['METTS MARK']))

cnt=0
for p_id, p_info in enron_data.items():
    if p_info['poi'] == 1:
        print("\nPerson ID:", p_id)
        cnt+=1

    #for key in p_info:
    #    print(key + ':', p_info[key])

print("Number of POI:{0}".format(cnt))

stockval = enron_data["PRENTICE JAMES"]["total_stock_value"]
print("What is the total value of the stock belonging to James Prentice? {0}".format(stockval))

num_emails_from_this_person_to_poi = enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print("How many email messages do we have from Wesley Colwell to persons of interest? {0}".format(num_emails_from_this_person_to_poi))

stock_options = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print("What’s the value of stock options exercised by Jeffrey K Skilling? {0}".format(stock_options))
