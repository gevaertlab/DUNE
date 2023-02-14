# %%
import os
from os.path import join
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score, as_concordance_index_ipcw_scorer
from tqdm import tqdm
import argparse
from scipy.stats import entropy
from termcolor import colored
import warnings
import joblib


# %%

warnings.simplefilter(action='ignore', category=FutureWarning)
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
    config["load_models"] = eval(config["load_models"])

    return config


def create_dependencies(model_dir):

    output_dirs = join(model_dir, "multivariate", "models")
    os.makedirs(output_dirs, exist_ok=True)

    return join(model_dir, "multivariate")


def create_fulldataset(csv_paths, metadata_path):

    metadata = pd.read_csv(metadata_path, index_col="eid")

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


def create_train_test_datasets(merged, var, task):

    features = merged[[k for k in merged.columns if k.isdigit()]]
    if task != "survival":
        labels = merged[var]
        labels = np.array(labels)
    else:
        time = merged[f"{var}_delay"].astype(float)
        event = merged[f"{var}_event"].astype(bool)
        labels = np.array([(e, t) for e, t in zip(event, time)], dtype=[
                          ('Status', '?'), ('Survival_in_days', '<f8')])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=12)

    return X_train, y_train, X_test, y_test


# %%
if __name__ == '__main__':

    np.random.seed(333)
    config = parse_arguments()

    # ########
    # os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    # config = configparser.ConfigParser()
    # config.read("/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining/UNet_6b_8f_UKfull/config/multivariate.cfg")
    # config = dict(config["config"])
    # config["model_path"] = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining/UNet_6b_8f_UKfull"
    # #######

    output_dir = create_dependencies(config['model_path'])

    # Importing datasets
    model_path = config['model_path']
    metadata_path = config["metadata"]
    feature_paths = [join(model_path, "autoencoding/features",
                          f"{file}_features.csv") for file in ["train", "test"]]
    merged = create_fulldataset(feature_paths, metadata_path)

    # Importing task list
    tasks = pd.read_csv(config["variables"])
    tasks = tasks[tasks["keep_model"]]
    task_list = {k: task for k, task in zip(
        tasks["var"], tasks["task"])}
    variables = list(task_list.keys())

    # Preparing results dataframe
    results_df = tasks.copy()
    results_df['ibs'] = np.nan

    for _, var in enumerate(bar := tqdm(variables, colour="green")):
        task = task_list[var]

        bar.set_description(colored(f"\n{var} - {task}", "yellow"))

        X_train, y_train, X_test, y_test = create_train_test_datasets(
            merged, var, task)
        n_feat = X_train.shape[1]

        num_classes = np.nan
        variance = np.nan
        scoring = None

        if task == "survival":
            lower, upper = np.percentile(merged["death_delay"], [10, 90])
            times = np.arange(lower, upper + 1)

            mod = RandomSurvivalForest(max_depth=1, random_state=123)
            mod = as_concordance_index_ipcw_scorer(mod, tau=times[-1])
            dict_params = {"estimator__n_estimators": np.arange(250, 500, 50),
                           "estimator__min_samples_split": np.arange(2, 10)
                           }
            # mod = CoxnetSurvivalAnalysis(fit_baseline_model=True)
            # dict_params = {"alpha_min_ratio": np.logspace(-6, -4, 3),
            #                "l1_ratio": np.linspace(0.005,1, 20)
            #                }
            # scoring = "ci"

        elif task == "regression":
            mod = Ridge(alpha=0.014)
            dict_params = {"alpha": np.logspace(-4, 1, 30)}
            variance = np.var(y_train)
            scoring = "r2"

        elif task == "classification":
            num_classes = int(len(set(y_train)))
            _, px = np.unique(y_train, return_counts=True)
            px = px / len(y_train)
            variance = entropy(px, base=2)

            mod = RidgeClassifier(alpha=0.014)
            dict_params = {"alpha":np.logspace(-4,1, 30)}
            # mod = RandomForestClassifier(random_state=123)
            # dict_params = {"n_estimators": np.arange(100, 500, 100),
                        #    "min_samples_split": np.arange(2, 5)}
            scoring = "f1_weighted"

        else:
            print("Unknown task.")
            break
        
        model_loaded = False
        if config["load_models"]:
            models_dir = join(output_dir, "models")
            try:
                mod = [m for m in os.listdir(models_dir) if var in m][0]
                mod = joblib.load(join(models_dir, mod))
                model_loaded = True
            except IndexError:
                pass

            # Use the whole dataset as test set
            X_test = pd.concat([X_train, X_test], axis=0)
            y_test = np.concatenate([y_train, y_test])
        
        if not model_loaded :
            mod = GridSearchCV(mod, dict_params, scoring=scoring, cv=3, n_jobs=-1)
            mod.fit(X_train, y_train)
            mod = mod.best_estimator_

        res = mod.score(X_test, y_test)
        
        if task == "survival":
            surv_probs = np.row_stack([
                fn(times) for fn in mod.predict_survival_function(X_test)])
            ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
            results_df.loc[results_df['var'] == var, "ibs"] = round(ibs, 4)

        # Logging results
        model_type = str(type(mod)).split(".")[-1].replace("'>","")
        results_df.loc[results_df['var'] == var, "performance"] = round(res, 4)
        results_df.loc[results_df['var'] == var,
                       "variance"] = round(variance, 4)
        results_df.loc[results_df['var'] == var, "num_classes"] = num_classes
        results_df.loc[results_df['var'] == var, "model"] = model_type
        results_df.loc[results_df['var'] == var, "metric"] = scoring
        results_df.loc[results_df['var'] == var, "restored_model"] = model_loaded

        output_file = join(output_dir, f"0-{config['output_name']}.csv")
        results_df.to_csv(output_file, index=False)

        # Saving model
        if not model_loaded:
            model_name = f"{var}_{model_type}_{scoring}.sav"
            joblib.dump(mod, join(output_dir, "models", model_name))
