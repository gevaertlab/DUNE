import os
import pandas as pd
import numpy as np
from os.path import join
from multiprocessing import Pool
from functools import partial


ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
MODEL_DIR = "outputs/UNet/pretraining"
os.chdir(ROOT_DIR)
MAX_EPOCH = 1000 # 1000

def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs


def train_models(model, task_dict):
    print(model)
    for _, variable in enumerate(task_dict.keys()):

        try:
            current_epoch = pd.read_csv(join(MODEL_DIR, model, "predictions",variable,"report.csv")).shape[0]
        except FileNotFoundError:
            current_epoch = 0
        
        if MAX_EPOCH - current_epoch <= 0:
            num_epochs = 0
        else:
            num_epochs = MAX_EPOCH - current_epoch

        cmd = f"CUDA_LAUNCH_BLOCKING=1 \
        python aspects/0_MRI/predictions/train_pred.py \
                --variable {variable} --task {task_dict[variable]} \
                --config {MODEL_DIR}/{model}/config/predict.json \
                --num_epochs {num_epochs}" 

        os.system(cmd)

def summarize_AE(autoencoder):
    """
    creates a summary.csv with performances of basic NNs to predict clinical parameters based on features extracted by the AE
    """

    ae_dir = os.path.join(MODEL_DIR, autoencoder, "predictions")
    variables_results = dict.fromkeys(ls_dir(ae_dir))

    for var in variables_results.keys():
        try:
            report = pd.read_csv(os.path.join(ae_dir, var, "report.csv"))
            if "val_r2_score" in report.columns:
                variables_results[var] = report['val_r2_score'].iloc[-1]
            else:
                variables_results[var] = report['val_accuracy'].iloc[-1]
        except:
            variables_results[var] = np.nan
            pass       


    ae_sum = pd.DataFrame.from_dict(
        variables_results, orient="index").reset_index()
    ae_sum['model'] = autoencoder
    ae_sum = ae_sum.rename(
        columns={"index": "var", 0: "performance"})
    
    variables = pd.read_csv("data/metadata/UKB_variables.csv")
    ae_sum = pd.merge(ae_sum, variables, how="left", on="var")
    
    ae_sum.to_csv(f"{ae_dir}/0-multivariate_results.csv", index=False)
    
    return ae_sum



def main():


    list_of_AEs = [
        "UNet_5b_4f_UKfull",
        "UNet_5b_8f_UKfull",
        "UNet_6b_4f_UKfull",
        "UNet_6b_8f_UKfull"
        ]

    task_dict = {
        # "age": "regression",
        # "sex": "classification",
        # "sleep": "regression",
        # "height" : "regression",
        # "weight" : "regression",
        # "bmi" : "regression",
        # "diastole" : "regression",
        # "smoking" : "classification",
        "alcohol_freq" : "regression",
        "greymat_vol" : "regression",
        "brain_vol" : "regression",
        "norm_brainvol": "regression",
        # "fluidency" : "regression",
        # "digits_symbols" : "regression",
        # "depression" : "classification",
        # "survival": "survival",
        # "survival_bin": "survival"
    }

    with Pool(8) as pool:
        train_func = partial(train_models, task_dict=task_dict)
        pool.map(train_func, list_of_AEs)


    overall = pd.DataFrame()
    for autoencoder in list_of_AEs:
        if autoencoder.startswith("UNet"):
            summary = summarize_AE(autoencoder)
            overall = pd.concat([overall, summary], axis=0)

    
    # exporting results
    overall = pd.DataFrame(overall, columns = ["var","task","group","performance", "model"])
    overall.to_csv(f"{MODEL_DIR}/multi_summary.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"\nFinished.")



