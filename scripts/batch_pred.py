import os
import pandas as pd
from os.path import join

MODEL_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet"
MAX_EPOCH = 500

def ls_dironly(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs
    


def train_models(list_of_models, task_dict, epoch_dict):

    for _, model in enumerate(list_of_models):

        for _, variable in enumerate(task_dict.keys()):

            try:
                current_epoch = pd.read_csv(join(MODEL_DIR, model, "predictions",variable,"report.csv")).shape[0]
            except FileNotFoundError:
                current_epoch = 0

            cmd = f"CUDA_LAUNCH_BLOCKING=1 \
            python aspects/0_MRI/predictions/train_pred.py \
                    --variable {variable} --task {task_dict[variable]} \
                    --config outputs/UNet/{model}/config/predict.json \
                    --num_epochs {MAX_EPOCH - current_epoch}" 

            os.system(cmd)

def main():
    ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOT_DIR)

    list_of_models = [
        "UNet_5b_4f_UKfull",
        "UNet_5b_8f_UKfull",
        "UNet_6b_4f_UKfull",
        "UNet_6b_8f_UKfull"
        ]

    task_dict = {
        "age": "regression",
        "sex": "classification",
        "sleep": "regression",
        "height" : "regression",
        "weight" : "regression",
        "bmi" : "regression",
        "diastole" : "regression",
        "smoking" : "classification",
        "alcohol_freq" : "classification",
        "alcohol_status" : "classification",
        "greymat_vol" : "regression",
        "brain_vol" : "regression",
        "norm_brainvol": "regression",
        "fluidency" : "regression",
        "digits_symbols" : "regression",
        "depression" : "classification",
        # "survival": "survival",
        # "survival_bin": "survival"
    }
    common_num_epochs = 100
    epoch_dict = {
        "age": common_num_epochs,
        "sex": common_num_epochs,
        "sleep": common_num_epochs,
        "height" : common_num_epochs,
        "weight" : common_num_epochs,
        "bmi" : common_num_epochs,
        "diastole" : common_num_epochs,
        "smoking" : common_num_epochs,
        "alcohol_freq" : common_num_epochs,
        "alcohol_status" : common_num_epochs,
        "greymat_vol" : common_num_epochs,
        "brain_vol" : common_num_epochs,
        "norm_brainvol": common_num_epochs,
        "fluidency" : common_num_epochs,
        "digits_symbols" : common_num_epochs,
        "depression" : common_num_epochs,
        # "survival": common_num_epochs,
        # "survival_bin": common_num_epochs
    }

    train_models(list_of_models, task_dict, epoch_dict)


if __name__ == "__main__":
    main()
    print(f"\nFinished.")
