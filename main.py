import preprocess
import sampling
import info_score
import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------

name = "data.csv"
data_file = "data"
#preprocess.inner_join(['Code', 'Year'], data_file, name)

df = pd.read_csv(name)
print(df.columns)

# -------------------------------------------------------------------

target = ["Life expectancy"]
predictor = ['CO2 emissions per capita',
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
             'Share of the population with access to electricity']

# c:= continuous, d:= discrete
target_discrete = np.full(len(target), False)
predictor_discrete = np.full(len(predictor), False)
predictor_discrete[7] = True

column_name = np.concatenate((target, predictor))
column_discrete = np.concatenate((target_discrete, predictor_discrete))

# -------------------------------------------------------------------

plot_file = 'data plot'

#preprocess.plot_hist(df[predictor], plot_file)
#preprocess.plot_hist(df[target], plot_file)
#preprocess.plot_scatter(df[target], df[predictor], plot_file)

# -------------------------------------------------------------------
# rolling cross validation

sampling_file = 'sampling'
test_train_fraction = 0.25
k = 5

#sampling.time_series_cv_split(df['Year'], test_train_fraction, k, sampling_file)

cv_test = []
cv_train = []

for each in os.listdir(sampling_file):
    for i in range(0, k):
        name_search = "cv_" + str(i)
        dir_str = os.path.join(sampling_file, each)
        try:
            dir_data = pd.read_csv(dir_str).to_numpy().flatten()
            if each == name_search + "_train.csv":
                cv_train.append(dir_data)
            elif each == name_search + "_test.csv":
                cv_test.append(dir_data)
        except:
            exit('csv issue')
# -------------------------------------------------------------------

info_score_file = 'info score'

info_index = pd.MultiIndex.from_product([range(0, k), target, predictor])

info_val = ['corr', 'mic_3']
info = pd.DataFrame(index=info_index, columns = info_val)

for i in range(0, 1): # fix k
    train_set = df.iloc[cv_train[i]]

    for each in info_val:
        each_index = info.loc[i].index.to_numpy()
        if each == 'corr':
            info.loc[i,each] = info_score.corr_coef(each_index, train_set)
        elif each.startswith('mic'):
            try:
                k_nn = int(each.split('_')[1])
                info.loc[i,each] = info_score.mic(each_index,
                                                  train_set,
                                                  column_name,
                                                  column_discrete,
                                                  k_nn)

            except:
                exit('mic issue')



