# %%
from os.path import join
import joblib
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from termcolor import colored
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV

from utils_ae import parse_arguments
from utils_pred import *


warnings.simplefilter(action='ignore', category=FutureWarning)
plt.switch_backend('agg')



def initialize_model(task, var, restore_models):

    assert task in ["survival","classification","regression"]

    if task == "survival":
        mod, hyperparams = init_survival_predictor(var, restore_models)
        scoring = None

    elif task == "regression":
        mod, hyperparams = init_regressor(var, restore_models)
        scoring = "r2"

    elif task == "classification":
        mod, hyperparams = init_classifier(var, restore_models)
        scoring = "f1_weighted"

    else:
        mod, hyperparams, scoring = None, None, None
    
    return mod, hyperparams, scoring


def cross_validate(X_train, y_train, X_test, y_test):

    num_classes, variance = None, None
    if task == "regression":
        num_classes, variance = None, np.var(y_train)
    elif task == "classification":
        num_classes, variance = calc_entropy(y_train)

    mod, hyperparams, scoring = initialize_model(task, var, restore_models)

    # Model training
    if not restore_models:
        mod = GridSearchCV(mod, hyperparams, scoring=scoring, cv=3, n_jobs=-1)
        mod.fit(X_train, y_train)
        mod = mod.best_estimator_

    else:
        # Use the whole dataset as test set
        X_test = pd.concat([X_train, X_test], axis=0)
        y_test = np.concatenate([y_train, y_test])

    # Scoring
    res = mod.score(X_test, y_test)    
    ibs = compute_brier(mod, X_test, y_test,  y_train) if task == "survival" else None

    # Saving model
    model_type = str(type(mod)).split(".")[-1].replace("'>","")
    if not restore_models:
        model_name = f"{var}_{split+1}_{model_type}_{scoring}.sav"
        joblib.dump(mod, join(output_dir, "models", model_name))

    # Return results
    results = {"variable":var,
            "split":split+1,
            "performance": round(res,4),
            "metric": scoring,
            "ibs": ibs,
            "variance": variance,
            "num_classes": [num_classes],
            "missing_rate": missing_rate,
            "model": model_type,
            "restored": restore_models
            }
    
    return results



# %%
if __name__ == '__main__':

    np.random.seed(334)
    config = parse_arguments("pred")


    # Importing datasets
    model_path = config['model_path']
    output_dir = join(config['model_path'] , "multivariate")
    output_file = join(output_dir, f"{config['output_name']}.csv")
    merged = create_fulldataset(**config)

    # Importing task list
    variables = join(config["data_path"], config["dataset"], "metadata", config["variables"])
    variables = pd.read_csv(variables)
    variables = variables.query("keep_model")
    variables = {k: task for k, task in zip(variables["var"], variables["task"])}

    # Create empty results
    results_df = pd.DataFrame()

    for _, var in enumerate(bar := tqdm(variables.keys(), colour="yellow")):
        bar.set_description(colored(f"\n{var}", "yellow"))
        task = variables[var]
        restore_models = config["load_models"]

        # Create subdataset
        features, labels , missing_rate = create_var_datasets(merged, var, task)

        if type_of_target(labels) in ["binary","multiclass"]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        else:
            skf = KFold(n_splits=5, shuffle=True, random_state=1)

        # Cross-validation loop
        for split, (train_index, test_index) in enumerate(skf.split(features, labels)):
            X_train = features[train_index]
            y_train = labels[train_index]
            X_test = features[test_index]
            y_test = labels[test_index]

            results = cross_validate(X_train, y_train, X_test, y_test)

            # Exporting results
            results = pd.DataFrame.from_dict(results, orient="columns")
            results_df = pd.concat([results_df, results], ignore_index=True)
            results_df.to_csv(output_file, index=False)
