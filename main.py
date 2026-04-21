import sampling
import preprocess
import info_score
import model
import selection

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------------

name = "data.csv"
data_file = "data"

'''
preprocess.inner_join(['Code', 'Year'], data_file, name)
'''

df = pd.read_csv(name)
print(df.columns)

# ----------------------------------------------------------------------------------------

target = np.array(["Life expectancy"])
predictor = np.array(['CO2 emissions per capita',
                      'Daily calorie supply per person',
                      'GDP per capita',
                      'Measles, first dose (MCV1)',
                      'Polio (Pol3)',
                      'Tetanus (DTP3)',
                      'Military expenditure (% of GDP)',
                      'Political regime',
                      'Population, total',
                      'Share of the population using the Internet',
                      'Share of the population using at least basic sanitation',
                      'Share of the labor force employed in agriculture',
                      'Share of the population with access to electricity'])

target_discrete = np.full(len(target), False)
predictor_discrete = np.full(len(predictor), False)
predictor_discrete[7] = True

column_name = np.concatenate((target, predictor))
column_discrete = np.concatenate((target_discrete, predictor_discrete))

# ----------------------------------------------------------------------------------------

plot_file = 'data/plot'

'''
preprocess.plot_hist(df[predictor], plot_file)
preprocess.plot_hist(df[target], plot_file)
preprocess.plot_scatter(df[target], df[predictor], plot_file)
'''

# ----------------------------------------------------------------------------------------
# rolling cross validation

sampling_file = 'sampling'
test_train_fraction = 0.25
k = 5

'''
sampling.time_series_cv_split(df['Year'], test_train_fraction, k, sampling_file)
'''

cv_test = []
cv_train = []

for each in os.listdir(sampling_file):
    for i in range(0, k):
        name_search = "cv_" + str(i)

        dir_str = os.path.join(sampling_file, each)
        dir_data = pd.read_csv(dir_str).to_numpy().flatten()

        if each == name_search + "_train.csv":
            cv_train.append(dir_data)
        elif each == name_search + "_test.csv":
            cv_test.append(dir_data)

# ----------------------------------------------------------------------------------------
info_name = "info/"

info_index = pd.MultiIndex.from_product([range(0, k), target, predictor])

'''
mi_c_k = [4, 6, 9, 13, 18, 24, 31, 39]
mi_m_k = [4]

#--------------
info_corr = pd.DataFrame(index=info_index, columns = ['corr'])
for cv_idx in range(0, k):
    train_set = df.loc[cv_train[cv_idx]]

    pair_label = info_corr.loc[cv_idx, 'corr'].index.to_frame(index = False).to_numpy()
    info_corr.loc[cv_idx, 'corr'] = info_score.compute('corr', pair_label, train_set)

#--------------
info_mi_c = pd.DataFrame(index=info_index, columns = mi_c_k)
for cv_idx in range(0, k):
    for k_val in mi_c_k:
        train_set = df.loc[cv_train[cv_idx]]

        pair_label = info_mi_c.loc[cv_idx, k_val].index.to_frame(index = False).to_numpy()
        info_mi_c.loc[cv_idx, k_val] = info_score.compute('mi,c_ksg,' + str(k_val), pair_label, train_set,
                                                          column_name, column_discrete)

#--------------
info_mi_m = pd.DataFrame(index=info_index, columns = mi_m_k)
for cv_idx in range(0, k):
    for k_val in mi_m_k:
        train_set = df.loc[cv_train[cv_idx]]

        pair_label = info_mi_m.loc[cv_idx, k_val].index.to_frame(index=False).to_numpy()
        info_mi_m.loc[cv_idx, k_val] = info_score.compute('mi,m_ross,' + str(k_val), pair_label, train_set,
                                                          column_name, column_discrete)

#--------------

info_corr.to_csv(info_name + "corr.csv")
info_mi_c.to_csv(info_name + "mic_c.csv")
info_mi_m.to_csv(info_name + "mic_m.csv")
'''

try:
    info_corr = pd.read_csv(info_name + "corr.csv")
    info_corr = info_corr.set_index(info_corr.columns[:3].tolist())

    info_mi_c = pd.read_csv(info_name + "mic_c.csv")
    info_mi_c = info_mi_c.set_index(info_mi_c.columns[:3].tolist())

    info_mi_m = pd.read_csv(info_name + "mic_m.csv")
    info_mi_m = info_mi_m.set_index(info_mi_m.columns[:3].tolist())
except:
    raise ValueError("info file issue")

# ----------------------------------------------------------------------------------------

info_plot_file = 'info/plot'

'''
for cv_idx in range(0, k):
    preprocess.plot_scatter(info_corr.loc[cv_idx], info_mi_c.loc[cv_idx], info_plot_file,
                            head = 'cv_' + str(cv_idx))
'''

info_mi_c['median'] = info_mi_c.median(axis=1)

# ----------------------------------------------------------------------------------------

schedule_folder = 'model/workspace/schedule/'

'''
for cv_idx in range(0, k):
    for each_target in target:
        model.scheduler.main(predictor, each_target,
                             info_corr.loc[cv_idx, 'corr'],
                             info_mi_c.loc[cv_idx, 'median'],
                             schedule_folder,
                             head = each_target, tail = '_cv_' + str(cv_idx))
'''

# ----------------------------------------------------------------------------------------

result_folder = 'model/workspace/result/'

'''
for cv_idx in range(0, k):
    print("-------")
    print("cv_idx " + str(cv_idx) )
    for each_target in target:
        print("---")
        print(each_target)
        try:
            schedule_each = pd.read_csv(schedule_folder + each_target + "_cv_" + str(cv_idx)+ ".csv")
        except:
            raise ValueError("schedule file issue")
        model.compiler.process(schedule_each, result_folder,
                               df.loc[cv_train[cv_idx]], df.loc[cv_test[cv_idx]],
                               head= each_target, tail='cv_' + str(cv_idx))

'''

# ----------------------------------------------------------------------------------------

result = []
minimization = []

for cv_idx in range(0, k):
    for each_target in target:
        result_name = result_folder + each_target + "_result_cv_" + str(cv_idx)+ ".csv"
        try:
            result_each = pd.read_csv(result_name)
            result.append(result_each)
        except:
            print("result file issue")
            print(cv_idx)
            print(each_target)

        minimization_name = result_folder + each_target + "_minimization_cv_" + str(cv_idx)+ ".txt"
        minimization.append(np.loadtxt(minimization_name))


result_plot = "model/workspace/plot/"
'''
for cv_idx in range(0, k):
    preprocess.plot_hist(result[cv_idx][['test error', 'skew', 'kurtosis','time']],
                         result_plot, tail = 'cv_' + str(cv_idx))
    preprocess.plot_scatter(result[cv_idx][['test error']],
                            result[cv_idx][['dof']],
                            result_plot, tail = 'cv_' + str(cv_idx))
    

for each_target in target:
    for cv_idx in range(0, k):
        each_txt = np.loadtxt(result_folder + each_target + "_minimization_cv_" + str(cv_idx) + ".txt")
        plt.hist(each_txt)
        plt.savefig(result_plot + each_target + "_minimization_cv_" + str(cv_idx) + ".png")
        plt.close()

'''

# ----------------------------------------------------------------------------------------

selection_folder = 'selection/'

chosen_model_per_cv = []
selection_col = None

'''
for each_target in target:
    for cv_idx in range(0, k):
        model_id = selection.min_dof_within_std(result[cv_idx][['id','dof','test error']])[0]

        schedule_dir = schedule_folder + each_target + "_cv_" + str(cv_idx) + ".csv"
        schedule_file = pd.read_csv(schedule_dir)

        if selection_col is None:
            selection_col = np.append(schedule_file.columns.to_numpy(), "cv")

        each_para = schedule_file[schedule_file['id'] == model_id].to_numpy().flatten()


        chosen_model_per_cv.append(np.append(each_para, cv_idx))

    chosen_model = pd.DataFrame(chosen_model_per_cv, columns=selection_col)
    chosen_model.to_csv(selection_folder + each_target + "_schedule.csv")

'''

# ----------------------------------------------------------------------------------------

selection_result = 'selection/result/'

'''
for each_target in target:
    for cv_idx in range(0, k):
        chosen_model = pd.read_csv(selection_folder + each_target + "_schedule.csv")
        model.compiler.process(chosen_model, selection_result,
                               df.loc[cv_train[cv_idx]], df.loc[cv_test[cv_idx]],
                               head = each_target,
                               tail = 'cv_' + str(cv_idx))

'''

# ----------------------------------------------------------------------------------------

chosen_overal = pd.DataFrame(index = range(k), columns = ["id", "dof"])

for each_target in target:
    for cv_idx in range(0, k):
        string_name = selection_result + each_target + "_result_cv_" + str(cv_idx) + ".csv"
        per_cv_result = pd.read_csv(string_name)

        chosen_overal.iloc[cv_idx] = selection.min_dof_within_std(per_cv_result)

    print("---------")
    print(each_target)
    print(chosen_overal)
    chosen_overal.to_csv('selection/' + each_target + "_result.csv")

# ----------------------------------------------------------------------------------------


result_merged = pd.concat(result, keys = range(k), names = ['cv_idx'])
minimization_merged = np.concatenate(minimization)


model_fit_time = result_merged['time'].to_numpy()

percentile = [50, 75, 90, 95]

print("---------")
print("model fitting time")

for each_per in percentile:
    print(str(each_per) + "th: " + str(np.percentile(model_fit_time, each_per)) + " seconds")

print("---------")
print("minimization fitting time")

for each_per in percentile:
    print(str(each_per) + "th: " + str(np.percentile(minimization_merged, each_per)) + " seconds")

print("---------")
print("minimization / model fitting")

for each_per in percentile:
    print(str(each_per) + "th: " + str(np.percentile(minimization_merged, each_per)/np.percentile(model_fit_time, each_per)))

