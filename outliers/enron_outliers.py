#!/usr/bin/python

import pickle
import numpy as np
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
# Outliers, Quiz 17: Remove the known outliers.
data_dict.pop( "TOTAL", 0 )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

data
### your code below

print(data_dict['METTS MARK'])
max_salary = np.sort(data[:,0])
print(max_salary[-2])

cnt=0
for p_id, p_info in data_dict.items():
    if (float(p_info['salary']) > 1000000.0) & (float(p_info['bonus']) > 5000000.0):
        print("Bandit: {0}".format(p_id))
    if p_info['poi'] == 1:
        # print("\nPerson ID:", p_id)
        cnt+=1


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
