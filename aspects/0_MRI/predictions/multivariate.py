import os
from os.path import join
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import learning_curve
from tqdm import tqdm
import argparse


plt.switch_backend('agg')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        help='model_path')
    args = parser.parse_args()

    config_file = join(args.model_path, "config/multivariate.cfg")

    config = configparser.ConfigParser()
    config.read(config_file)
    config = dict(config["config"])
    config["model_path"] = args.model_path

    return config


def create_dependencies(model_dir):

    output_dir = join(model_dir, "multivariate")
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def evaluate(model, X_train, y_train, X_test, scoring, output_dir, name="var", printing=False):

    N, train_score, val_score = learning_curve(
        model, X_train, y_train, cv=4,
        scoring=scoring,
        train_sizes=np.linspace(.1, 1, 30)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if printing:
        plt.figure()
        plt.title(name)
        plt.plot(N, train_score.mean(axis=1), label="train_score")
        plt.plot(N, val_score.mean(axis=1), label="val_score")
        plt.legend()
        plt.ylabel(scoring)
        plt.savefig(join(output_dir, name+".png"))

    return y_pred


def create_fulldataset(csv_paths, metadata_path):

    metadata = pd.read_excel(metadata_path, index_col="eid")

    list_features = []
    for f in csv_paths:
        df = pd.read_csv(f)
        list_features.append(df)
    features = pd.concat(list_features, axis=0, ignore_index=True)
    features["eid"] = features["Unnamed: 0"]
    features = features.drop(["Unnamed: 0"], axis=1).set_index("eid")

    merged = metadata.merge(features, how="inner",
                            left_index=True, right_index=True)

    return merged


def create_train_test_datasets(merged, variable):

    features = merged[[k for k in merged.columns if k.isdigit()]]
    labels = merged[variable]
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=12)

    return X_train, y_train, X_test, y_test


def main():

    np.random.seed(333)
    config = parse_arguments()
    output_dir = create_dependencies(config['model_path'])

    # Importing datasets
    model_path = config['model_path']
    metadata_path = config["metadata"]
    feature_paths = [join(model_path, "autoencoding/features",
                          f"{file}_features.csv") for file in ["train", "test"]]
    merged = create_fulldataset(feature_paths, metadata_path)

    variables = [v for v in merged.columns if not v.isdigit()]
    task_list = pd.read_csv(config["variables"])
    task_list = {k: task for k, task in zip(
        task_list["newname"], task_list["task"])}

    # Multiple variables testing

    results_df = pd.read_excel(metadata_path, index_col="eid")
    results_df = results_df[results_df['var'].isin(variables)]
    results_df['res'] = np.nan

    results = {}
    for var in tqdm(variables, colour="green"):
        X_train, y_train, X_test, y_test = create_train_test_datasets(
            merged, var)

        if task_list[var] == "regression":
            XGB = XGBRegressor(tree_method='gpu_hist', gpu_id=0)
            scoring = "r2"
        else:
            XGB = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
            scoring = "f1_weighted"

        y_pred = evaluate(XGB, X_train, y_train,
                          X_test, scoring, output_dir, var, printing=True)

        res = np.nan
        if task_list[var] == "regression":
            res = r2_score(y_test, y_pred)
        else:
            res = accuracy_score(y_test, y_pred)

        results_df.loc[results_df['var'] == var, "res"] = round(res, 4)

        output_file = join(output_dir, "0-multivariate.csv")
        results_df.to_csv(output_file)


if __name__ == '__main__':
    main()
