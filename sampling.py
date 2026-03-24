import numpy as np
import pandas as pd

def time_series_cv_split(time_id, test_train_fraction, k, path):

    sorted_idx = np.argsort(time_id.to_numpy())

    nrow = time_id.shape[0]

    for i in range(0, k):
        portion = (i+1)/k
        portion_count = round(portion * nrow)

        train_frac = portion/(1+test_train_fraction)
        train_count = round(train_frac*nrow)
        train_index = sorted_idx[0:train_count]

        test_index = sorted_idx[train_count:portion_count]

        train_out = pd.DataFrame({'train_set': train_index})
        test_out = pd.DataFrame({'test_set': test_index})

        name_out = path + "/cv_" + str(i)
        train_out.to_csv(name_out + "_train.csv", index=False)
        test_out.to_csv( name_out + "_test.csv", index=False)

    return