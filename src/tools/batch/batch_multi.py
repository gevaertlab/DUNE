import os
import pandas as pd
from os.path import join
import re

MODEL_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning"

def ls_dironly(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs
    


def multivariate(list_of_models):

    for _, model in enumerate(list_of_models):
        print("Evaluating ", model)
        cmd = f"python src/predictions/multivariate.py --model_path {MODEL_DIR}/{model}"

        os.system(cmd)

def main():
    ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOT_DIR)

    list_of_models = [
        # "6b_4f_REMB",
        # "6b_4f_REMB_segm",
        "6b_4f_TCGA",
        "6b_4f_TCGA_segm"#,
        # "6b_4f_UCSF",
        # "6b_4f_UCSF_segm",
        # "6b_4f_UPENN",
        # "6b_4f_UPENN_segm"
        ]

    multivariate(list_of_models)

    overall = pd.DataFrame()
    for model in list_of_models:
        try:
            mod_summ = pd.read_csv(join(MODEL_DIR, model, "multivariate","0-last.csv"))
            b, f = re.findall(r'\d+', model)
            mod_summ['model_'] = model


            overall = pd.concat([overall, mod_summ])
        except FileNotFoundError:
            pass    
    overall.to_csv(MODEL_DIR + "/0-synth_multi.csv", index=False)



if __name__ == "__main__":
    main()
    print(f"\nFinished.")
