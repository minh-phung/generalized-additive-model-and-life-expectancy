import numpy as np
import pandas as pd
import time
import sklearn.feature_selection as fs

def corr_coef(index_pair, data):
    cor_vec = np.full(len(index_pair), np.nan)
    for i, each in enumerate(index_pair):
        if len(each) != 2:
            exit("corr index pair issue")
        else:
            x = data[each[0]]
            y = data[each[1]]
            cor_vec[i] = np.corrcoef(x, y)[0, 1]

    return cor_vec

def mic(index_pair, data, variable, variable_discrete, k_nn):

    mic_vec = np.full(len(index_pair), np.nan)
    for i, each in enumerate(index_pair):
        if len(each) != 2:
            exit("mic index pair issue")
        else:
            x = data[each[0]]
            x_index = np.where(variable == each[0])[0][0]
            x_discrete = variable_discrete[x_index]

            y = data[each[1]]
            y_index = np.where(variable == each[1])[0][0]
            y_discrete = variable_discrete[y_index]





    return