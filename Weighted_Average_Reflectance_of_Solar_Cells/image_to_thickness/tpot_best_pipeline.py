import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('tpot_best_pipeline.csv').drop(columns=['Unnamed: 0'])
train, test = train_test_split(tpot_data, test_size=0.3, random_state=42)
training_features, training_target = train.iloc[:,:4], train.iloc[:,4]
testing_features, testing_target = test.iloc[:,:4], test.iloc[:,4]

# Average CV score on the training set was: -8.673579471970266
exported_pipeline = make_pipeline(
    PCA(iterated_power=5, svd_solver="randomized"),
    AdaBoostRegressor(learning_rate=0.5, loss="exponential", n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)