from get_data import read_params
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

''' Split data into train and test set. '''

def split_data(config_path):
    config = read_params(config_path)
    raw_data_path = config["preprocessed_data"]["preprocessed_data_csv"]
    label_path = config["load_data"]["train_labels"]
    df = pd.read_csv(raw_data_path)
    labels = []
    X = df
    file1 = open(label_path,"r")
    for i in file1:
        labels.append(i)
    train_label = [x[:-1] for x in labels]

    X['status'] = train_label
    test_data_path = config["split_data"]["test_path"] 
    train_data_path = config["split_data"]["train_path"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    # Split data into train and test set
    train, test = train_test_split(X, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")
    



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_data(config_path=parsed_args.config)