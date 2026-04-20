import numpy as np

def min_dof_within_std(data, std_dof_match = False):

    sorted = data.sort_values(by='test error', ascending=True)


    test_error_std = sorted['test error'].std()

    test_error_min = sorted['test error'].min()

    within = data[data['test error'] < test_error_min + test_error_std]

    chosen_dof = within.loc[within['dof'].idxmin()]

    return np.array([chosen_dof['id'], chosen_dof['dof']])

