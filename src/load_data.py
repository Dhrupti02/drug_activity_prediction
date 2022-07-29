# read the data from data source
# save it in the data/raw for further process
import os
from get_data import read_params, get_data
import argparse

''' This will load data from the given path in the config_path. '''

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    train = df[0]
    labels = df[1]
    new_cols = [col for col in train.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    train_labels = config["load_data"]["train_labels"]
    train.to_csv(raw_data_path, sep=",", index=False, header=new_cols)
    with open(train_labels, 'w') as f:
        for line in labels:
            f.write(f"{line}\n")
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()                    
    load_and_save(config_path=parsed_args.config)      