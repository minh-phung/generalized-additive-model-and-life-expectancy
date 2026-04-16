import numpy as np
import pandas as pd
from pygam import GAM, l, s, LinearGAM
from functools import reduce
from operator import add
import matplotlib.pyplot as plt

from .model import main

def process(schedule, result_folder,
            train, test, head = '', tail = ''):

    nrow = schedule.shape[0]
    columns = ["id", "dof", "test error", "skew", "kurtosis", "time"]

    out = pd.DataFrame(index=range(nrow*2), columns=columns)
    mini_solver_time = np.array(np.zeros(nrow))

    issue = []

    for i, row in schedule.iterrows():
        #'''
        try:
            print("model id " + str(row['id']))
            each = main(row, test, train)

            out.iloc[i] = each[0][0]
            out.iloc[i+nrow] = each[0][1]

            mini_solver_time[i] = each[1]
        except:
            issue.append(i)
        #'''
        '''
        if i == 0:
            print(main(row, test, train))
        '''
    print("issue")
    print(issue)

    out.to_csv(result_folder + head + "_result_" + tail + ".csv",
               index = False)

    return

