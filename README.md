From ourworldindata.org, multiple time-series variables were chosen, based on two criterias:
- Area of research: Categorized from the source, variables were chosen to cover all areas, with at least one variable in each
- Country and time stamp overlap: Given that variables will be inner joined on country and time stamp (year), some were filtered out in order to not be the bottleneck in terms of final observation count. The exact final observation count is >2000.

data.csv contains all inner joined data that is used.

The chosen target is "life expectancy" (continuous), with all others being predictors (12 continuous + 1 discrete). 

Histograms of all variates, and scatter plot of variates vs. target are plotted and saved in 'data/plot'.

Given that the data is time series, rolling cross-validation is used. The data is sorted chronologically and chosen to be divided into k = 5 equal sections (section 0 to 4). Cross validation set i includes up the section i (for i = 2, section 0, 1, 2 are included). Within each cross validation set, the chronologically first 75% of the entries are the training set, the last 25% are the testing set). Index of training and testing sets, for each cross validation set i (from 0 to 4) is in 'sampling'.

The generalized linear model will have two types of terms: linear and cubic splines. In order to avoid overfitting, two approaches are taken:
- Feature selection: For each pair of predictor and target, (Pearson) linear correlation and mutual information are computed on each cross validation set. For mutual information, continuous - continuous pairs use Kraskov–Stögbauer–Grassberger 2003 (infomeasure), and continuous - discrete pairs use Ross 2014 (sklearn). The result of the computation is saved in 'info'. The selection process for each cross validation set is as follows:
  - Looking at the absolute value of linear correlation in descending order, one family of model is chosen where the top half are linear terms, and the bottom half are spline terms. More families of model are chosen by varying where the linear/spline separation is (by 3 variate), in both direction.
  - Per family of model, up to half of variates with the lowest mutual information (continuous - continuous) is removed from the model. Since there is only one pair of discrete - continuous mutual information, one variation is with and the other is without.
  - Each variation on each family of models is listed in 'model/workspace/schedule', identified by the respective cross validation index.
- Smoothing: Spline terms are far more likely to overfit than linear terms. As a result, two variations are fitted per model: one unconstrained and one constrained. The constrained variation is imposed by having a non-zero lambda on all spline terms. With the defaul of having degree 3 polynomial (cubic) and a total of 6 basis funcion (2 knots), the lambda is chosen such that would reduce the degrees of freedom, on each variate, by a fraction of 0.75. Solving for the lambda requires a minimization step (scipy.optimize.minimize)

The result of all models is saved in 'model/workspace/result', identified by the respective cross validation index.

