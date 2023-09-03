import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Disable the specific warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import BayesianRidge, ARDRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from tpot import TPOTRegressor

# Hyperparameter Tuning
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV

import pickle
import json
import os
import errno


# fitting 시간 측정용
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time() - ts
        print(f"Time consumed loading/training model :: {te:.2f} s")
        return result

    return timed


# 성능평가함수
def perf_eval_fc(y_pred, y_test):
    r2_score_val = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAE = mean_absolute_error(y_test, y_pred)
    MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return r2_score_val, MSE, RMSE, MAE, MAPE


# 모델저장
def save_model(model, model_name):
    pickle.dump(model, open(model_name, "wb"))


# 모델로딩
def load_model(model_name):
    return pickle.load(open(model_name, "rb"))


# 예측값 저장
def save_prediction(y_pred, model_name):
    y_pred = pd.DataFrame(y_pred, columns=["Predicted Thickness"])
    y_pred.to_csv(model_name, index=False)


# 성능평가표 업데이트
def update_table(perf_table, y_pred, y_test, model_name):
    r2_score_val, MSE, RMSE, MAE, MAPE = perf_eval_fc(y_pred, y_test)
    perf_table.loc[model_name] = [r2_score_val, RMSE, MSE, MAE, MAPE]
    return perf_table


# 디렉토리 생성
def make_dir(path):
    try:
        if not (os.path.isdir(path)):
            os.makedirs(os.path.join(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise


# 그래프 그리기
def graphize(y_test, y_pred):
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
    sns.regplot(
        x=y_test,
        y=y_pred,
        scatter_kws={"s": 20, "alpha": 0.3},
        line_kws={"color": "green", "linewidth": 2},
        robust=True,
    )

    plt.xlabel("Actual Thickness")
    plt.ylabel("Predicted Thickness")
    plt.title("Actual vs Predicted Thickness (Test Set)")
    plt.show()


# 모델 fitting
@timeit
def fit_models(
    X_train, y_train, model_name, model, model_path, hyperparams=None, cv=None
):
    try:
        opt = load_model(model_path + model_name + ".pkl")
        print("Model found. Loading the model >> " + model_name)
    except:
        print("Model not found. Training the model >> " + model_name)
        if hyperparams is None:
            opt = model
            opt.fit(X_train, y_train)
        else:
            try:
                opt = BayesSearchCV(
                    model,
                    hyperparams,
                    cv=cv,
                    n_iter=50,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    random_state=42,
                )
                opt.fit(X_train, y_train)
            except:
                print("BayesSearchCV failed. Using the grid search >> " + model_name)
                opt = GridSearchCV(
                    model,
                    hyperparams,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
                opt.fit(X_train, y_train)

        save_model(opt, model_path + model_name + ".pkl")
    return opt


class Solar_pipeline(object):
    """Solar_pipeline

    Parameters
    ----------
    dataset : pandas.DataFrame
        전처리된 데이터셋
    target : str
        타겟 변수명
    hyperparams_path : str
        하이퍼파라미터가 저장된 json 파일 경로
    model_path : str
        모델이 저장될 경로
    pred_path : str
        예측값이 저장될 경로
    test_size : float
        테스트셋 비율
    random_state : int
        랜덤시드

    Methods
    -------
    data_split()
        데이터셋을 훈련셋과 테스트셋으로 분리
    read_hyperparams(hyperparams_path)
        하이퍼파라미터 json 파일을 읽어서 반환
    fit()
        모델 훈련
    display_perf_table(metric = 'RMSE')
        성능평가표 출력
    summarize(metric = 'RMSE')
        최적 모델 성능평가표 출력 및 예측값 시각화

    """

    def __init__(
        self,
        dataset,
        target,
        hyperparams_path,
        model_path,
        pred_path,
        except_model=None,
        tpot=False,
        cv=5,
        test_size=0.3,
        random_state=42,
    ):
        self.dataset = dataset
        self.target = target
        self.hyperparams = self.read_hyperparams(hyperparams_path)
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.except_model = except_model
        self.tpot = tpot
        # dir 생성
        make_dir(model_path)
        make_dir(pred_path)
        self.model_path = model_path
        self.pred_path = pred_path

        self.X_train, self.X_test, self.y_train, self.y_test = self.data_split()
        self.perf_table = pd.DataFrame(columns=["R2", "RMSE", "MSE", "MAE", "MAPE"])
        self.best_models = {}
        np.random.seed(random_state)

    def data_split(self):
        X = self.dataset.drop(self.target, axis=1)
        y = self.dataset[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def read_hyperparams(self, hyperparams_path):
        with open(hyperparams_path, "r") as f:
            hyperparams = json.load(f)
        return hyperparams

    def display_dataset(self, head=5):
        return self.dataset.head(head)

    def fit(self, save_pred=True):
        """
        Tpot 사용 시
        """
        if self.tpot == True:
            tpot = TPOTRegressor(
                generations=5,
                population_size=50,
                verbosity=2,
                random_state=self.random_state,
                cv=self.cv,
                n_jobs=-1,
            )
            tpot.fit(self.X_train, self.y_train)
            self.best_models["tpot"] = tpot.fitted_pipeline_
            y_pred = self.best_models["tpot"].predict(self.X_test)
            if save_pred == True:
                save_prediction(y_pred, self.pred_path + "tpot.csv")
            self.perf_table = update_table(self.perf_table, y_pred, self.y_test, "tpot")
            print(self.perf_table.loc["tpot"])
            return
        """
        Tpot 사용 X 시
        """
        for model_name, model in zip(
            [
                "LinearRegression",
                "Ridge",
                "Lasso",
                "ElasticNet",
                "BayesianRidge",
                "ARDRegression",
                "SGDRegressor",
                "SVR",
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "AdaBoostRegressor",
                "XGBRegressor",
                "LGBMRegressor",
                "CatBoostRegressor",
                "KNeighborsRegressor",
                "DecisionTreeRegressor",
                "MLPRegressor",
            ],
            [
                LinearRegression(),
                Ridge(),
                Lasso(),
                ElasticNet(),
                BayesianRidge(),
                ARDRegression(),
                SGDRegressor(),
                SVR(cache_size=7000),
                RandomForestRegressor(),
                GradientBoostingRegressor(),
                AdaBoostRegressor(),
                XGBRegressor(),
                LGBMRegressor(),
                CatBoostRegressor(verbose=False),
                KNeighborsRegressor(),
                DecisionTreeRegressor(),
                MLPRegressor(),
            ],
        ):
            if self.except_model != None and model_name in self.except_model:
                print("Except model >> " + model_name)
                continue
            self.best_models[model_name] = fit_models(
                self.X_train,
                self.y_train,
                model_name,
                model,
                self.model_path,
                hyperparams=self.hyperparams[model_name],
                cv=self.cv,
            )
            try:
                print(
                    "BEST PARAMS >> "
                    + model_name
                    + ": "
                    + str(self.best_models[model_name].best_params_)
                )
            except:
                print("No hyperparameters to tune >> " + model_name)

            print("*" * 50)
            print("\n")
            y_pred = self.best_models[model_name].predict(self.X_test)
            if save_pred == True:
                save_prediction(y_pred, self.pred_path + model_name + ".csv")

            self.perf_table = update_table(
                self.perf_table, y_pred, self.y_test, model_name
            )
            print(self.perf_table.loc[model_name])

    def display_perf_table(self, metric="RMSE", ascending=True):
        return round(self.perf_table.sort_values(by=metric, ascending=ascending), 5)

    def summarize(self, metric="RMSE"):
        if self.tpot == True:
            Best_model = self.best_models["tpot"]
            y_pred = Best_model.predict(self.X_test)
            best_model_name = "tpot"

        else:
            Best_model = self.perf_table.sort_values(by=metric, ascending=True).index[0]
            print("Best Model: " + Best_model)
            print("Best hyperparams: " + str(self.best_models[Best_model].best_params_))
            print(
                "Best performance: "
                + str(self.perf_table.sort_values(by=metric, ascending=True).iloc[0])
            )
            best_model_name = Best_model
            Best_model = load_model(self.model_path + Best_model + ".pkl")
            y_pred = Best_model.predict(self.X_test)

        # 그래프 그리기
        graphize(self.y_test, y_pred)

        test_pred = pd.DataFrame(
            {"Actual Thickness": self.y_test, "Predicted Thickness": y_pred}
        )

        return Best_model, best_model_name, y_pred, test_pred
