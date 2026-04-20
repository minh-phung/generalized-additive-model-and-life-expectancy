import numpy as np
import pandas as pd
from pygam import GAM, l, s, LinearGAM
from functools import reduce
from operator import add
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import time


def edof_per_term(gam):
    edof_per_coef = gam.statistics_['edof_per_coef']
    edof_per_term = np.full(len(gam.terms) - 1, 0.1)
    per_term_idx = 0

    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        indices = gam.terms.get_coef_indices(i)
        edof_per_term[per_term_idx] = edof_per_coef[indices].sum()
        per_term_idx += 1

    return edof_per_term

def fit_get_edof(lam, X, y, terms):
    gam = LinearGAM(terms, lam=lam).fit(X, y)
    per_term = edof_per_term(gam)
    #print(per_term)
    return per_term

def constrain_edof_fraction(X, y, terms,
                            full_edof, fraction,
                            default_lam=None):
    term_list = [t for t in terms if not t.isintercept]

    if default_lam is None:
        default_lam = np.zeros(len(term_list))

    # -----------------------------------

    spline_idx = [i for i, t in enumerate(term_list) if t.n_coefs > 1]

    target_edof = full_edof.copy()
    target_edof[spline_idx] = full_edof[spline_idx] * fraction
    # -----------------------------------

    def loss(log_lam_spline):

        lam_spline = np.exp(log_lam_spline)

        lam = default_lam.copy()
        lam[spline_idx] = lam_spline

        edof = fit_get_edof(lam, X, y, terms)

        loss_val = np.sum((edof[spline_idx] - target_edof[spline_idx]) ** 2)

        return loss_val

    bounds = [(-10, 10)] * len(spline_idx)
    res = minimize(
        loss, x0 = np.zeros(len(spline_idx)),
        tol = 1e-2
    )

    lam_spline = np.exp(res.x)

    out_lam = default_lam.copy()
    out_lam[spline_idx] = lam_spline

    return out_lam

# ---------------------------------------------------------------

def info_output(gam, X_test, y_test):

    dof = sum(edof_per_term(gam))

    residual_test  = y_test - gam.predict(X_test)

    test_error = (sum(residual_test**2))**(1/2) / X_test.shape[0]

    skew_test = skew(residual_test)
    kurt_test = kurtosis(residual_test)

    return dof, test_error, skew_test, kurt_test

# ---------------------------------------------------------------

def main(parameter, test, train):

    para_col =  parameter.index.values

    term = []

    x_col = []
    x_idx = 0

    y_col = np.nan

    for i, each_col in enumerate(para_col):
        if parameter[each_col] == 'l' or parameter[each_col] == 's':
            if parameter[each_col] == 'l':
                term.append(l(x_idx))
            else:
                term.append(s(x_idx, n_splines= 6, spline_order= 3))
            x_col.append(each_col)
            x_idx += 1

        elif parameter[each_col] == 't':
            y_col = each_col

    X_train = train[x_col].to_numpy()
    y_train = train[y_col].to_numpy()

    X_test = test[x_col].to_numpy()
    y_test = test[y_col].to_numpy()

    terms = reduce(add, term)

    if np.linalg.matrix_rank(X_train) != x_idx:
        print("X_train rank issue")

    # ----------------------------------------------
    full_start = time.time()
    gam_full = LinearGAM(terms, lam = 0).fit(X_train, y_train)
    #gam_full.summary()
    full_time = time.time() - full_start

    info_full = info_output(gam_full, X_test, y_test)
    # ----------------------------------------------
    mini_solv_start = time.time()
    lam_constrained = constrain_edof_fraction(
        X_train, y_train, terms, edof_per_term(gam_full), 0.75
    )
    mini_solv_time = time.time() - mini_solv_start

    # ------------------------
    const_start = time.time()
    gam_const = LinearGAM(terms, lam = lam_constrained).fit(X_train, y_train)
    #gam_const.summary()
    const_time = time.time() - const_start

    info_const = info_output(gam_const, X_test, y_test)
    #print(info_const)
    # ----------------------------------------------

    out = np.array([ [parameter['id'], *info_full, full_time],
                    [parameter['id'], *info_const, const_time] ])

    return out, mini_solv_time

