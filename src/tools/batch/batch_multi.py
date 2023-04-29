import os
import pandas as pd
from os.path import join
from tqdm import tqdm
import re

MODEL_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/"

def ls_dironly(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs
    


def main():
    ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOT_DIR)

    list_cond = [
        ["AE", "AE_UCSF_segm", "whole_brain", "UCSF_wholebrain"],
        ["AE", "AE_TCGA_segm", "whole_brain", "TCGA_wholebrain"],
        ["AE", "AE_UPENN_segm", "whole_brain", "UPENN_wholebrain"],

        ["AE", "AE_UCSF_segm", "tumor", "UCSF_tumor_only"],
        ["AE", "AE_TCGA_segm", "tumor", "TCGA_tumor_only"],
        ["AE", "AE_UPENN_segm", "tumor", "UPENN_tumor_only"],

        ["AE", "AE_UCSF_segm", "combined", "UCSF_combined"],
        ["AE", "AE_TCGA_segm", "combined", "TCGA_combined"],
        ["AE", "AE_UPENN_segm", "combined", "UPENN_combined"],
  
        ["AE", "AE_UCSF_segm", "radiomics", "UCSF_radiomics"],
        ["AE", "AE_TCGA_segm", "radiomics", "TCGA_radiomics"],
        ["AE", "AE_UPENN_segm", "radiomics", "UPENN_radiomics"]
  
        ]

    overall = pd.DataFrame()
    for cond in tqdm(list_cond, colour="red"):
        folder, model, features, output_name = cond
        model_dir = join(MODEL_DIR, folder)
        cmd = f"python src/predictions/multivariate.py --model_path {model_dir}/{model} --features {features} --output_name {output_name}"

        os.system(cmd)


        try:
            mod_summ = pd.read_csv(join(model_dir, model, "multivariate",f"{output_name}.csv"))
            # b, f = re.findall(r'\d+', model)
            mod_summ['AE'] = model
            mod_summ['features'] = features

            overall = pd.concat([overall, mod_summ])
        except FileNotFoundError:
            pass    
        overall.to_csv(MODEL_DIR + "tumor_crop.csv", index=False)



if __name__ == "__main__":
    main()
    print(f"\nFinished.")
