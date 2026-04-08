import numpy as np
import pandas as pd

def process(schedule, checkpoint_folder, result_folder,
            train, test, head = '', tail = ''):

    for i, row in schedule.iterrows():
        print(row)
        model(row, test, train)


        break

    return

def model(parameter, test, train):

    para_col =  parameter.columns.values

    term = []
    for each_col in para_col:
        if parameter[each_col] == 'l':
            print(each_col)


    return