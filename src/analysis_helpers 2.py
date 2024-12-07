import numpy as np

def count_mean(row):
    '''
    Function which takes in a row consisting of a list and calculates the mean ignoring nan values
    '''
    return np.nanmean(row)