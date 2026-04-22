From ourworldindata.org, multiple time-series variables were chosen, based on two criterias:
- Area of research: Categorized from the source, variables were chosen to cover all areas, with at least one variable in each.
- Country and time stamp overlap: Given that variables will be inner joined on country and time stamp (year), some were filtered out in order to not be the bottleneck in terms of final observation count. The exact final observation count is >2000.

data.csv contains all inner joined data that is used.

The chosen target is "life expectancy" (continuous), with all others being predictors (12 continuous + 1 discrete). 

Histograms of all variates, and scatter plot of variates vs. target are plotted and saved in 'data/plot'.

Given that the data is time series, rolling cross-validation is used. The data is sorted chronologically and chosen to be divided into k = 5 equal sections (section 0 to 4). Cross validation set i includes up the section i (for i = 2, section 0, 1, 2 are included). Within each cross validation set, the chronologically first 75% of the entries are the training set, the last 25% are the testing set. Index of training and testing sets, for each cross validation set i (from 0 to 4), is in 'sampling'.

The generalized additive model (pygam) will have two types of terms: linear and cubic splines. In order to avoid overfitting, two approaches are taken:
- Feature selection: For each pair of predictor and target, (Pearson) linear correlation and mutual information are computed on each cross validation set. For mutual information, continuous - continuous pairs use Kraskov–Stögbauer–Grassberger (KSG) 2003 (infomeasure), and continuous - discrete pairs use Ross 2014 (sklearn). Since KSG utilizes nearest-neighbour, a variety of choices for neighbourhood size were used, per cross validation. The result of the computation is saved in 'info'. Plots of linear correlation vs. mutual information is presented in 'info/plot'. The overal median across neighbourhood sizes was then computed. The selection process for each cross validation set is as follows:
  - Looking at the absolute value of linear correlation in descending order, one family of model is chosen where the top half are linear terms, and the bottom half are spline terms. More families of model are chosen by varying where the linear/spline separation is (by 3 variate), in both direction.
  - Per family of model, up to half of variates with the lowest mutual information (continuous - continuous) are removed from the model. Since there is only one pair of discrete - continuous mutual information, one variation is with and the other is without.
  - Each variation on each family of models is listed in 'model/workspace/schedule', identified by the respective cross validation index.
- Smoothing: Spline terms are far more likely to overfit than linear terms. As a result, two variations are fitted per model: one unconstrained and one constrained. The constrained variation is imposed by having non-zero lambdas on all spline terms. With the defaul of having degree 3 polynomial (cubic) and a total of 6 basis function (2 knots), each lambda is chosen to be such that would reduce the degrees of freedom, on each variate, by a fraction of 0.75. Solving for the lambdas require a minimization step (scipy.optimize.minimize).

Per model, the variables that are computed are degree of freedom, test error, skewness, and kurtosis. The results of all models are saved in 'model/workspace/result', identified by the respective cross validation index. Plots of test error vs. degree of freedom, as well as histograms of all variables mentioned, per cross validation, are saved in 'model/workspace/plot'. Records of time elapsed to run each model, as well as time takes to run each minimization steps, are also recorded.

Per cross validation set, the chosen model is the one the the least degrees of freedom, that is within one standard deviation (of tests errors with all degrees of freedom within the set) of the minimum test error. There are overal 5 chosen models, each on their respective cross validation set, saved in 'selection/Life expectancy_schedule.csv'.

The 5 models are then fitted again, this time on all 5 cross validation sets. The result is saved under 'selection/result'. The same exact overal model selection process is used on each cross validation set. It is saved in 'selection/result.csv'.

Out of 5 cross validation sets, one specific model was chosen 4/5 sets, with the constrained variation 3/4 sets. By referencing the id with the schedule, in 'selection/Life expectancy_schedule.csv', one can get the exact choice for linear, spline, and removed terms.

Looking at the time it takes to fit a model, the 95th percentile is less than 0.2 seconds. However, in comparing the appropriate percentile with the time it takes to compute the minimization step (for finding lambdas) (percentile time of minimization step / percentile of the modeling step), at the 50th, it is almost a factor of 50. At the 95th, it is more than 100.


References:
Hastie et al. (2001). Elements of Statistical Learning.;
Murphy, Kevin. (2012). Machine Learning - A Probabilistic Perspective.
