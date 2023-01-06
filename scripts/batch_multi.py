import os
import pandas as pd
from os.path import join

MODEL_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining"

def ls_dironly(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs
    


def multivariate(list_of_models):

    for _, model in enumerate(list_of_models):
        print("Evaluating ", model)
        cmd = f"python aspects/0_MRI/predictions/multivariate.py \
		--model_path {MODEL_DIR}/{model}"

        os.system(cmd)

def main():
    ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOT_DIR)

    list_of_models = [
        "UNet_5b_4f_UKfull",
        "UNet_5b_8f_UKfull",
        "UNet_6b_4f_UKfull"d,
        "UNet_6b_8f_UKfull"
        ]

    # multivariate(list_of_models)

    overall = pd.DataFrame()
    for model in list_of_models:
        mod_summ = pd.read_csv(join(MODEL_DIR, model, "multivariate","0-multivariate.csv"))
        mod_summ['model'] = model

        overall = pd.concat([overall, mod_summ])
    
    overall.to_csv(MODEL_DIR + "/multi_summary.csv", index=False)



if __name__ == "__main__":
    main()
    print(f"\nFinished.")
