import os
import pandas as pd
from os.path import join
from multiprocessing import Pool
from functools import partial


ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
MODEL_DIR = "outputs/UNet/final"
os.chdir(ROOT_DIR)
MAX_EPOCH = 60 # 1000

def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs


def train_model(variable, task_dict, model):
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


def main():

    model = "UNet_5b4f_TCGA"
    task_dict = {
        "IDH1_bin":"classification",
        "IDH1":"classification",
        "survival": "survival",
        "survival_bin": "survival",
        "grade_binary": "classification"
    }

    tasks = task_dict.keys()

    with Pool(8) as pool:
        train_func = partial(train_model, task_dict=task_dict, model=model)
        pool.map(train_func, tasks)



if __name__ == "__main__":
    main()
    print(f"\nFinished.")



