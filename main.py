import preprocess
import sampling
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
target_label    = np.full(len(target), "c")
predictor_label = np.full(len(predictor), "c")
predictor_label[7] = 'd'

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

cv_test  = []
cv_train = []

for each in os.listdir(sampling_file):
    for i in range(0, k):
        name_search = "cv_" + str(i)
        dir_str = os.path.join(sampling_file, each)
        if each == name_search + "_train.csv":
            cv_train.append(dir_str)
        elif each == name_search + "_test.csv":
            cv_test.append(dir_str)

# -------------------------------------------------------------------

