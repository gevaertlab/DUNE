# %%
from os.path import join
import joblib
import warnings
from tqdm import tqdm
import joblib
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from termcolor import colored
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle
import seaborn as sns

from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

from utils_ae import parse_arguments
from utils_pred import *


import sys
import os
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


plt.switch_backend('agg')


def initialize_model(task, var, restore_models, path):

    assert task in ["survival", "classification", "regression"]

    if task == "survival":
        mod, hyperparams = init_survival_predictor(var, restore_models, path)
        scoring = None

    elif task == "regression":
        mod, hyperparams = init_regressor(var, restore_models, path)
        scoring = "r2"

    elif task == "classification":
        mod, hyperparams = init_classifier(var, restore_models, path)
        scoring = "f1_weighted"

    else:
        mod, hyperparams, scoring = None, None, None

    return mod, hyperparams, scoring


def export_roc_curve():

    num_classes = len(pred_probas[0])

    gt = np.array(ground_truth)
    pb = np.array(pred_probas)
    df = np.array(decision_functions)

    fig, ax = plt.subplots(figsize=(4, 4))

    if num_classes == 2:
        RocCurveDisplay.from_predictions(
            gt,
            df,
            name=f"ROC curve for class 1",
            ax=ax
        )
    else:
        gt = label_binarize(ground_truth, classes=list(set(ground_truth)))
        colors = cycle(["aqua", "forestgreen", "darkturquoise",
                       "gold", "darkorange", "orangered"])
        for class_id, color in zip(range(num_classes), colors):
            RocCurveDisplay.from_predictions(
                gt[:, class_id],
                pb[:, class_id],
                name=f"ROC curve for class {class_id}",
                color=color,
                ax=ax,
            )

    ax.set_title(var)
    plt.close()

    return fig


def export_km_curves():

    event, time = zip(*ground_truth)
    event, time = np.array(event), np.array(time)
    group = np.where(predictions > np.median(predictions), "high", "low")

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    high_risk = (group == "high")

    pval = logrank_test(time[high_risk], time[~high_risk],
                        event[high_risk], event[~high_risk]).p_value

    fig, ax = plt.subplots(figsize=(4, 5))
    kmf_high.fit(time[high_risk], event[high_risk], label="High risk")
    kmf_low.fit(time[~high_risk], event[~high_risk], label="Low risk")

    kmf_high.plot_survival_function(c="darkred", show_censors=True, ax=ax)
    kmf_low.plot_survival_function(c="darkblue", show_censors=True, ax=ax)
    add_at_risk_counts(kmf_high, kmf_low, ax=ax)

    ax.set_xlabel("Time from inclusion (days)")
    ax.set_ylabel("Survival")
    ax.set_title(var)
    ax.text(x=0, y=0.12, s=f"p={pval:.3f}")
    fig.tight_layout()

    plt.close()

    return fig


def export_cm():
    feat = config["features"]

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(ground_truth, predictions)
    cm = sns.heatmap(data=cm, annot=True, fmt=',d', ax=ax)
    cm.set(xlabel='Predicted', ylabel='Truth',
           title=f"{var}/{feat}_{split+1}/{NUM_SPLITS}")
    # plt.savefig(join(output_dir, f"conf_mat/cm_{var}_{feat}.png"))
    plt.close()

    return fig


def export_scatter():

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(
        ground_truth,
        predictions, s=1
    )

    ax.plot(np.linspace(-3, 4, 500), np.linspace(-3, 4, 500), "-r")

    ax.set_xlabel("ground_truth")
    ax.set_ylabel("prediction")
    ax.set_title(var)

    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)

    plt.close()

    return fig


def export_results(var, pred_probas, decision_functions):

    if not pred_probas:
        pred_probas = np.nan
        decision_functions = np.nan

    df = pd.DataFrame(
        {f"{var}__gt": ground_truth,
         f"{var}__pred": predictions,
         f"{var}__pb": pred_probas,
         f"{var}__df": decision_functions},
         )

    return df


def export_model(mod, metric):
    joblib.dump(mod, "RF_survival")


def run_split(X_train, y_train, X_test, y_test):

    num_classes, variance = None, None
    if task == "regression":
        num_classes, variance = None, np.var(y_train)
    elif task == "classification":
        num_classes, variance = calc_entropy(y_train)

    mod, hyperparams, scoring = initialize_model(
        task, var, restore_models, path=output_dir)

    # Model training
    if not restore_models:
        mod = GridSearchCV(mod, hyperparams, scoring=scoring, cv=3, n_jobs=-1)
        mod.fit(X_train, y_train)
        mod = mod.best_estimator_

    # Scoring
    res = mod.score(X_test, y_test)
    ibs = compute_brier(mod, X_test, y_test,
                        y_train) if task == "survival" else None

    # Return results
    model_type = str(type(mod)).split(".")[-1].replace("'>", "")
    results = {"variable": var,
               "split": split+1,
               "performance": round(res, 4),
               "metric": scoring,
               "ibs": ibs,
               "variance": variance,
               "num_classes": [num_classes],
               "N": int(X_test.shape[0]),
               "missing_rate": missing_rate,
               "model": model_type,
               "restored": restore_models,
               }

    return mod, results


# %%
if __name__ == '__main__':

    np.random.seed(334)
    config = parse_arguments("pred")

    # Importing datasets
    model_path = config['model_path']
    output_dir = join(config['model_path'], "multivariate")
    output_file = join(output_dir, f"{config['output_name']}.csv")
    all_var_predictions_file = join(output_dir, f"{config['output_name']}_PREDICTIONS.csv.gz")
    output_plots = join(output_dir, f"{config['output_name']}_plots.pdf")
    merged = create_fulldataset(**config)

    # Importing task list
    variables = join(config["data_path"], config["dataset"],
                     "metadata", config["variables"])
    variables = pd.read_csv(variables)
    variables = variables.query("keep_model")
    variables = {k: task for k, task in zip(
        variables["var"], variables["task"])}

    # Create empty results
    results_df = pd.DataFrame()
    allvar_predictions = pd.DataFrame()
    pp = PdfPages(output_plots)

    for _, var in enumerate(bar := tqdm(variables, colour="yellow")):
        bar.set_description(colored(f"\n{var}", "yellow"))
        task = variables[var]
        restore_models = config["load_models"]
        predictions, pred_probas, decision_functions, ground_truth = [], [], [], []
        performances = []
        
        # identifiers = []

        # Create subdataset
        features, labels, missing_rate = create_var_datasets(merged, var, task)

        NUM_SPLITS = 5
        if task == "classification" and type_of_target(labels) in ["binary", "multiclass"]:
            skf = StratifiedKFold(n_splits=NUM_SPLITS,
                                  shuffle=True, random_state=1)
        else:
            skf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=1)

        # Cross-validation loop
        for split, (train_index, test_index) in enumerate(skf.split(features, labels)):
            X_train = features[train_index]
            y_train = labels[train_index]
            X_test = features[test_index]
            y_test = labels[test_index]

            mod, results = run_split(X_train, y_train, X_test, y_test)
            ground_truth.extend(y_test)
            predictions.extend(mod.predict(X_test))
            # identifiers.extend(idx[test_index])

            performances.append((mod, results["performance"]))
            if task == "classification":
                decision_functions.extend(mod.decision_function(X_test))
                pred_probas.extend(mod.predict_proba(X_test))

            # Exporting results
            results = pd.DataFrame.from_dict(results, orient="columns")
            results_df = pd.concat([results_df, results], ignore_index=True)
            results_df.to_csv(output_file, index=False)

            if split == NUM_SPLITS-1:
                if task == "classification":
                    pp.savefig(export_cm())
                    pp.savefig(export_roc_curve())
                elif task == "regression":
                    pp.savefig(export_scatter())
                elif task == "survival":
                    # export_model(mod, "survival")
                    pp.savefig(export_km_curves())

                var_predictions = export_results(var, pred_probas, decision_functions)
                allvar_predictions = pd.concat([allvar_predictions, var_predictions], axis=1)
                allvar_predictions["mod"] = config["output_name"]
                allvar_predictions.set_index("mod").to_csv(all_var_predictions_file, index=True)

    pp.close()
