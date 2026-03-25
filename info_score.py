import numpy as np
import pandas as pd
import time
import sklearn.feature_selection as fs

def corr_coef(index_pair, data):
    cor_vec = np.full(len(index_pair), np.nan)
    for i, each in enumerate(index_pair):
        x = data[each[0]]
        y = data[each[1]]
        cor_vec[i] = np.corrcoef(x, y)[0, 1]

    return cor_vec

def discrete_bin(data, bin_choice, parameter_0):

    if bin_choice == 'q':
        label = range(parameter_0)
        return pd.qcut(data, q = parameter_0, labels = label).to_numpy()

    return

def mic_discrete(index_pair, data,
                 variable_list = None, variable_discrete = None,
                 bin_choice = None, parameter_0 = None):

    for i, each_pair in enumerate(index_pair):
        print(each_pair)
        # ------------------------------------

        x = data[each_pair[0]].to_numpy()
        x_index = np.where(variable_list == each_pair[0])[0]
        x_discrete = variable_discrete[x_index]
        '''
        x_adjusted = np.full(len(x), np.nan)
        if x_discrete:
            x_adjusted = x
        else:
            print("x discretized")
            x_adjusted = discrete_bin(x, bin_choice, parameter_0)
        '''
        # ------------------------------------

        y = data[each_pair[1]].to_numpy()
        y_index = np.where(variable_list == each_pair[1])[0]
        y_discrete = variable_discrete[y_index]

        y_adjusted = np.full(len(y), np.nan)
        if y_discrete:
            y_adjusted = y
        else:
            print("y discretize")
            print(y)
            print(bin_choice)
            print(parameter_0)
            y_adjusted = discrete_bin(y, bin_choice, parameter_0)
            print(y_adjusted)
        # ------------------------------------


    return

def compute(value, index_pair, data,
            variable_list = None, variable_discrete = None):

    if index_pair.shape[1] != 2:
        exit("index pair issue")
    else:
        try:
            # -------------------------------------
            if value == 'corr':
                return corr_coef(index_pair, data)
        except:
            exit("corr issue")
            # -------------------------------------
        try:
            if value.startswith('mic'):
                if all(variable_discrete) :
                    return mic_discrete(index_pair, data,
                                        variable_list, variable_discrete)
                else:
                    mic_type = value.split("_")
                    print(mic_type)
                    return mic_discrete(index_pair, data,
                                        variable_list, variable_discrete,
                                        bin_choice = mic_type[1],
                                        parameter_0 = int(mic_type[2]))
        except:
            exit("mic issue")

    return