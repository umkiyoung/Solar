import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
dt_train = pd.read_csv("img_to_reflectance_train_data.csv")
dt_test = pd.read_csv("img_to_reflectance_test_data.csv")

training_features = dt_train.drop(columns=['Reflectance'],axis=1)
training_target = dt_train.iloc[:,-1]
testing_features = dt_test.drop(columns=['Reflectance'],axis=1)
testing_target = dt_test.iloc[:,-1]

# Average CV score on the training set was: -5.618899818609263
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.5, loss="huber", max_depth=3, max_features=0.9000000000000001, min_samples_leaf=20, min_samples_split=13, n_estimators=100, subsample=0.9500000000000001)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=6, min_samples_split=13)),
    RandomForestRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=8, min_samples_split=12, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
