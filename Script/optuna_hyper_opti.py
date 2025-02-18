from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import optuna
from sklearn_ml_class import CalML
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import logging
import sys
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def objective(trial):

    random = trial.suggest_int("random", 10, 100)
    # random = 88
    ML_calc = CalML(src, independent_vars_labels, dependent_vars_labels, random)
    
    # ————————opti hyper params for Catboost——————————
    if model_name == "Cat":
        param = {
            "loss_function": trial.suggest_categorical("loss_function", ["RMSE","MAE","MAPE","Poisson"]),
            # "loss_function": trial.suggest_categorical("loss_function", ["RMSE"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2, 10),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.001, 0.3),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered","Plain"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian","Bernoulli","MVS"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 10),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),      
        }
        model = CatBoostRegressor(**param)
        # model.fit(ML_calc.X_train.values, ML_calc.y_train.values, eval_set=[(ML_calc.X_test.values, ML_calc.y_test.values)], verbose=False, early_stopping_rounds=200)
        model.fit(ML_calc.X_train.values, ML_calc.y_train.values, verbose=False)

    # ————————opti hyper params for GPR——————————
    elif model_name == "GPR":
        param = {
                "noise": trial.suggest_float("noise", 0, 0.1),
                "nu": trial.suggest_float("nu", 0, 5),
                "constant_value": trial.suggest_float("constant_value", 0, 12),
                "scaler":trial.suggest_float("scaler", 1, 10),
        }
        noise = param["noise"]
        m52 = ConstantKernel(param["constant_value"])*Matern(length_scale=param["scaler"], nu=param["nu"])
        model = GaussianProcessRegressor(kernel=m52, alpha=noise**2)
        model.fit(ML_calc.X_train, ML_calc.y_train)

    # ——————————opti hyper params for RF——————————
    elif model_name == "RF":
        param = {
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"]),
                "n_estimators": trial.suggest_int("n_estimators",50, 300),
                "max_features": trial.suggest_categorical("max_features", ['auto','sqrt']),
                "max_depth": trial.suggest_int("max_depth", 10,500),
               "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
               "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "random_state":trial.suggest_int("random_state", 1, 100)
        }
        model = RandomForestRegressor(**param)
        model.fit(ML_calc.X_train,  np.ravel(ML_calc.y_train))

    # ——————————opti hyper params for SVR—————————— 
    elif model_name == "SVR":
        param = {
                "kernel": trial.suggest_categorical("kernel", ["poly","rbf","linear","sigmoid"]),
                "tol": trial.suggest_float("tol", 1e-4,1e-2),
                "C": trial.suggest_int("C", 1, 15),
                "epsilon": trial.suggest_float("epsilon", 0, 1),
        }
        if param["kernel"] == "poly":
            param["degree"] = trial.suggest_int("degree", 1, 6)

        model = SVR(**param)
        model.fit(ML_calc.X_train.values,  np.ravel(ML_calc.y_train.values))    

    # ——————————opti hyper params for KNN——————————
    elif model_name == "KNN":
        param = {
            'n_neighbors': trial.suggest_int('n_neighbors', 2, 10),
            'algorithm': trial.suggest_categorical('algorithm', ['auto','ball_tree','kd_tree','brute']),
            'weights': trial.suggest_categorical('weights', ['uniform','distance']),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60),
            "p": trial.suggest_categorical("p", [1, 2]),
        }

        model = KNeighborsRegressor(**param)
        model.fit(ML_calc.X_train.values,  ML_calc.y_train.values)     

    # ——————————opti hyper params for XGBoost——————————
    elif model_name == "XGB":
        param = {
            "n_estimators": trial.suggest_categorical("n_estimators", [5000,10000,15000,20000]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e0),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            "subsample": trial.suggest_categorical("subsample",[0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300)
        }    
        model = XGBRegressor(**param)
        model.fit(ML_calc.X_train.values, ML_calc.y_train.values, eval_set=[(ML_calc.X_test.values, ML_calc.y_test.values)], verbose=False, early_stopping_rounds=200)

    # ——————————opti hyper params for LGBM——————————
    elif model_name == "LGBM":
        param = {
            "n_estimators": trial.suggest_categorical("n_estimators", [5000,10000,15000,20000]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e0),
            "max_depth": trial.suggest_int("max_depth", 3, 30, step=3),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            "subsample": trial.suggest_categorical("subsample",[0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
            "num_leaves": trial.suggest_int("num_leaves", 21, 195, step=3),
            "num_iterations": trial.suggest_int("num_iterations", 10, 210, step=10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100, step=10),
        }    
        model = LGBMRegressor(**param)
        # model.fit(ML_calc.X_train.values, np.ravel(ML_calc.y_train.values))
        model.fit(ML_calc.X_train.values, np.ravel(ML_calc.y_train.values), eval_set=[(ML_calc.X_test.values, ML_calc.y_test.values)], verbose=False, early_stopping_rounds=500)

    # ————————————predict————————————
    y_pred = ML_calc.target_tt.inverse_transform(model.predict(ML_calc.X_test).reshape(-1,1))
    y_test = ML_calc.target_tt.inverse_transform(ML_calc.y_test.values.reshape(-1,1))

    MSE = mean_squared_error(y_test.reshape(-1,1), y_pred.reshape(-1,1))
    R2 = r2_score(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    # MSE plays better performance than R2 or RMSE
    return MSE

if __name__ == "__main__":

    independent_vars_labels = ["Ni", "Ti", "Fe"]
    dependent_vars_labels = ["CSI"]
    src = "../Dataset/NiTiFe-ML.csv"

    model_name = "Cat"
    property = dependent_vars_labels[0]

    study_name = "../model/%s_%s" % (model_name, property)

    if os.path.exists('../model/%s_%s_Ev.db' % (model_name, property)):
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.load_study(study_name=study_name, storage=storage_name)

    else:
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, sampler=TPESampler(), directions=["minimize"])

   
    study.optimize(objective, n_trials=500) # Run for 10 minutes

    print("Number of completed trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("\tBest Score: {}".format(trial.value))
    print("\tBest Params: ")


    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # default vislualization
    fig = optuna.visualization.plot_param_importances(study)
    # fig = optuna.visualization.plot_optimization_history(study)
    fig.show()