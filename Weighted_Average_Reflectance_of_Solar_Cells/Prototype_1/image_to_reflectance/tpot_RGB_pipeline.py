import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
dt_train = pd.read_csv("img_to_reflectance_train_data.csv")
dt_test = pd.read_csv("img_to_reflectance_test_data.csv")

training_features = dt_train.drop(columns=['R.I','Thickness','Reflectance'],axis=1)
training_target = dt_train.iloc[:,-1]
testing_features = dt_test.drop(columns=['R.I','Thickness','Reflectance'],axis=1)
testing_target = dt_test.iloc[:,-1]

# Average CV score on the training set was: -12.179621911784329
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            ),
            SelectPercentile(score_func=f_regression, percentile=91)
        ),
        FastICA(tol=0.15000000000000002)
    ),
    ExtraTreesRegressor(bootstrap=False, max_features=0.8500000000000001, min_samples_leaf=20, min_samples_split=7, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
