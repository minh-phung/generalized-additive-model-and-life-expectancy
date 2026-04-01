import numpy as np

import infomeasure as im
import sklearn.feature_selection as fs


def corr_coef(x, y,
              variable_x = None, variable_y = None):
    return np.corrcoef(x, y)[0, 1]

# -----------------------------------------------------------------------------

def mic_c_ksg(x, y, x_discrete, y_discrete, k_value,
              variable_x = None, variable_y = None):

    if not x_discrete and not y_discrete:
    # Kraskov–Stögbauer–Grassberger (KSG)
        mutual_info = im.estimator(x, y, measure = 'mi',
                                   approach = 'ksg', k = k_value).result()
        return mutual_info
    return np.nan

def mic_m_ross(x, y, x_discrete, y_discrete, k_value,
               variable_x = None, variable_y = None):

    if x_discrete and not y_discrete:
        return fs.mutual_info_classif(y.reshape(-1,1), x,
                                      discrete_features = y_discrete,
                                      n_neighbors= k_value)[0]
    elif not x_discrete and y_discrete:
        return fs.mutual_info_classif(x.reshape(-1,1), y,
                                      discrete_features = x_discrete,
                                      n_neighbors= k_value)[0]
    return np.nan

# -----------------------------------------------------------------------------

def compute(value, index_pair, data,
              variable_list = None, variable_discrete = None):

    if index_pair.shape[1] != 2:
        raise ValueError("Index pair shape issue")

    result = np.full(len(index_pair), np.nan)

    # ----------------------------------
    if value == 'corr':
        def func(x, y,
                 variable_x = None, variable_y = None):
            return corr_coef(x, y)

    elif value.startswith('mic'):
        # "mic,type,parameter" (symmetric)
        # type:
        # c := continuous - continuous
        # d := continuous - discrete

        def func(x, y,
                 variable_x = None, variable_y = None):
            try:
                _, mic_type, mic_param = value.split(',')
                mic_param = int(mic_param)
            except:
                raise ValueError("Invalid mic format")

            x_idx = np.where(variable_x == variable_list)[0]
            y_idx = np.where(variable_y == variable_list)[0]

            x_discrete = variable_discrete[x_idx]
            y_discrete = variable_discrete[y_idx]

            if mic_type == 'c_ksg':
                return mic_c_ksg(x, y, x_discrete, y_discrete, mic_param)
            elif mic_type == 'm_ross':
                return mic_m_ross(x, y, x_discrete, y_discrete, mic_param)
            return np.nan
    else:
        raise ValueError("Unknown info_val parameter")

    # ----------------------------------

    result = np.full(len(index_pair), np.nan)

    for i, (var1, var2) in enumerate(index_pair):
        x = data[var1].to_numpy()
        y = data[var2].to_numpy()

        result[i] = func(x, y, var1, var2)

    return result
