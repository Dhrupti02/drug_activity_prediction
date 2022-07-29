from get_data import read_params
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

import json
import os
import joblib
import logging

''' Train the data and evaluate the performance of the model. '''

# Performance matrices
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    ac = accuracy_score(actual, pred)
    return rmse, mae, ac


def train_and_evaluate(config_path):
    logger = logging.getLogger("train_and_evaluate")
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["AdaBoostClassifier"]["params"]["n_estimators"]
    learning_rate = config["estimators"]["AdaBoostClassifier"]["params"]["learning_rate"]
    # random_state = config["estimators"]["AdaBoostClassifier"]["params"]["random_state"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    ADB_model=AdaBoostClassifier(n_estimators = n_estimators, learning_rate = learning_rate)
    ADB_model.fit(train_x,train_y)
    y_pred_ada=ADB_model.predict(test_x)
    
    
    (rmse, mae, ac) = eval_metrics(test_y, y_pred_ada)
    logger.info('> RMSE: %.2f' % rmse)
    logger.info('> MAE: %.2f' % mae)
    logger.info('> Accuracy: %.2f' % ac)


    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "RMSE": rmse,
            "MAE": mae,
            "Accuracy": ac
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
               'n_estimators': n_estimators
        }
        json.dump(params, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(ADB_model, model_path)
    
# Create logger and assign handler
logger = logging.getLogger("train_and_evaluate")
handler  = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("train_and_evaluate.iter")
logger.setLevel(logging.INFO)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)