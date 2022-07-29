from get_data import read_params
import argparse
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA


# Project data on a reduced dimensionality k using PCA
def pca(config_path):

    config = read_params(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df = pd.read_csv(raw_data_path)
    n_components = config["base"]["n_components"]
    pca = sklearnPCA(n_components = n_components)
    pca.fit(df)
    PCA_projected_train = pca.transform(df)
    pca_data = pd.DataFrame(PCA_projected_train)
    raw_data_path = config["preprocessed_data"]["preprocessed_data_csv"]
    pca_data.to_csv(raw_data_path, sep=",", index=False, encoding="utf-8")



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    pca(config_path=parsed_args.config)