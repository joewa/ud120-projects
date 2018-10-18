#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    sqr_err = (predictions - net_worths)**2
    m = np.c_[ages, net_worths, sqr_err]
    # sort array with regards to 3rd column
    m = m[m[:,2].argsort()]
    # WARNING: returning a numpy array rather than a list of tuples

    # Take the 90% with the smallest error.
    cleaned_data = m[:int(len(m)*0.9)]

    return cleaned_data
