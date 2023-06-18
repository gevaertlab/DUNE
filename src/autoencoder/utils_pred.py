import os
from os.path import join
import joblib

import pandas as pd
import numpy as np
from scipy.stats import entropy


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, RidgeClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score


from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# %% Datasets
def create_fulldataset(**config):

    model_path = config['model_path'] 
    features = config["features"]

    features_path = str()
    if features == "TUM_radiomics":
        features_path = join(config["data_path"], config["dataset"], "metadata/0-TUM_pyradiomics.csv.gz")
    elif features == "WB_radiomics":
        features_path = join(config["data_path"], config["dataset"], "metadata/0-WB_pyradiomics.csv.gz")
    elif features == "whole_brain":
        features_path = join(model_path, "exports/features/whole_brain.csv.gz")
    elif features == "tumor":
        features_path = join(model_path, "exports/features/tumor.csv.gz")
    elif features == "combined":
        features_path = join(model_path, "exports/features/wb_and_radiomics.csv.gz")
    else:
        print("Invalid feature source")

    features = pd.read_csv(features_path).set_index("eid")

    if type(config["dataset"]) == list:
        metadata = config["metadata"]
    else:
        metadata = join(config["data_path"], config["dataset"], "metadata", config["metadata"])

    metadata = pd.read_csv(metadata, index_col="eid")

    merged = metadata.merge(features, how="inner",
                            left_index=True, right_index=True)

    return merged


def create_var_datasets(merged, var, task):

    nrow_init = merged.shape[0]

    if task != "survival":
        labels = merged[var]
        labels = np.array(labels)

        merged = merged.loc[~np.isnan(labels)]
        labels = labels[~np.isnan(labels)]
    else:
        time = merged[f"{var}_delay"].astype(float)
        event = merged[f"{var}_event"].astype(bool)
        labels = np.array([(e, t) for e, t in zip(event, time)], dtype=[
                          ('Status', '?'), ('Survival_in_days', '<f8')])

        merged = merged.loc[~np.isnan(time)]
        labels = labels[~np.isnan(time)]        

    features = merged[[k for k in merged.columns if k.isdigit()]]
    missing_rate = round(1-(features.shape[0] / nrow_init), 2)

    # scaling
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # resampling
    if task == "classification":
        sampler = RandomOverSampler(sampling_strategy="minority")
        features, labels = sampler.fit_resample(features, labels)


    return features, labels, missing_rate


# %% Models
def init_survival_predictor(var, restore=False, path=None):

    if restore:
        mod = load_model(var,path)
        hyperparams = {}

    else:
        mod = RandomSurvivalForest(max_depth=1, random_state=123)
        # mod = as_concordance_index_ipcw_scorer(mod)
        hyperparams = {
            "n_estimators": np.arange(200, 500, 100),
            "min_samples_split": np.arange(2, 10)}

    return mod, hyperparams

def init_classifier(var, restore=False, path=None):
    if restore:
        mod = load_model(var,path)
        hyperparams = None
    else:
        mod = RidgeClassifier(alpha=0.014)
        hyperparams = {"alpha":np.logspace(-4,1, 30)}

    return mod, hyperparams

def init_regressor(var, restore=False, path=None):
    if restore:
        mod = load_model(var,path)
        hyperparams = None

    else:
        mod = Ridge(alpha=0.014)
        hyperparams = {"alpha": np.logspace(-4, 1, 30)}
        
    return mod, hyperparams


def load_model(var, path):
    models_dir = join(path, "models")
    try:
        mod = [m for m in os.listdir(models_dir) if var in m][0]
        mod = joblib.load(join(models_dir, mod))
    except IndexError:
        pass

    return mod



# %% Misc

def calc_entropy(y_train):

    num_classes = int(len(set(y_train)))
    _, px = np.unique(y_train, return_counts=True)
    px = px / len(y_train)
    variance = round(entropy(px, base=2),4)

    return num_classes, variance

def compute_brier(mod, X_test, y_test,  y_train):   

    max_train_surv = np.max([[t[1] for t in y_train]])
    filt = [t[1]< max_train_surv for t in y_test]

    X_test = X_test[filt]
    y_test = y_test[filt]

    survivals = [t[1] for t in y_test]
    lower, upper = np.percentile(survivals, [10, 90])
    times = np.arange(lower, upper,10)

    surv_probs = np.row_stack([
    fn(times) for fn in mod.predict_survival_function(X_test)])
    ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
    
    return ibs


