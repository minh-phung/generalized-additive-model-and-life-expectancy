import os
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inner_join(key, path, out_name):
    csv_list = os.listdir(path)

    dfs = []

    for each in csv_list:
        full_path = os.path.join(path, each)
        data = pd.read_csv(full_path).dropna()
        if data[key].duplicated().any():
            exit("duplicate")

        dfs.append(data)

    df = dfs[0]
    for i, each in enumerate(dfs[1:], start = 2):
        df = df.merge(each, on=key, how='inner')

    df.to_csv(out_name, index=False)

    return

def plot_hist(data, path, tail = ''):

    columns = data.columns

    for each in columns:
        plt.hist(data[each], bins='auto')
        name = path + "/" + each + '_hist'
        if tail != '':
            name += "_" + tail
        plt.savefig(name)
        plt.close()

    return

def plot_scatter(target, predictor, path, tail = ''):

    col_pred = predictor.columns
    col_targ = target.columns

    for each_pred in col_pred:
        for each_targ in col_targ:
            plt.scatter(predictor[each_pred],
                        target[each_targ],
                        s = 10)
            plt.xlabel(each_pred)
            plt.ylabel(each_targ)
            name = path + "/" + each_pred + "_scatter"
            if tail != '':
                name += "_" + tail
            plt.savefig(name)
            plt.close()

    return

