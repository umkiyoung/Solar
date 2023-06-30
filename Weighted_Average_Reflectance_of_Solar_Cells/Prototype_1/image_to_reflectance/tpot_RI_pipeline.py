import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
dt_train = pd.read_csv("img_to_reflectance_train_data.csv")
dt_test = pd.read_csv("img_to_reflectance_test_data.csv")

training_features = dt_train.drop(columns=['Thickness','Reflectance'],axis=1)
training_target = dt_train.iloc[:,-1]
testing_features = dt_test.drop(columns=['Thickness','Reflectance'],axis=1)
testing_target = dt_test.iloc[:,-1]

# Average CV score on the training set was: -13.831829767511152
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    RandomForestRegressor(bootstrap=True, max_features=0.9500000000000001, min_samples_leaf=7, min_samples_split=5, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
