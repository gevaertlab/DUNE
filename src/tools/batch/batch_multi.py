import os
import pandas as pd
from os.path import join
from tqdm import tqdm
import re

MODEL_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/"

def ls_dironly(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs
    


def main():
    ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOT_DIR)

    list_cond = [

        ["6b_4f_UCSF_segm", "features", "UCSF_features"],
        ["6b_4f_UCSF_segm", "radiomics", "UCSF_radiomics"],
        ["6b_4f_UCSF_segm", "combined", "UCSF_combined"],

        ["6b_4f_UCSF_segm2", "features", "UCSF_features"],
        ["6b_4f_UCSF_segm2", "radiomics", "UCSF_radiomics"],
        ["6b_4f_UCSF_segm2", "combined", "UCSF_combined"],

        ["6b_4f_TCGA_segm","features", "TCGA_features"],
        ["6b_4f_TCGA_segm","radiomics", "TCGA_radiomics"],
        ["6b_4f_TCGA_segm","combined", "TCGA_combined"],

        ["6b_4f_UPENN_segm", "features", "UPENN_features"],
        ["6b_4f_UPENN_segm", "radiomics", "UPENN_radiomics"],
        ["6b_4f_UPENN_segm", "combined", "UPENN_combined"]

        ]

    overall = pd.DataFrame()
    for cond in tqdm(list_cond, colour="red"):
        model, features, output_name = cond

        cmd = f"python src/predictions/multivariate.py --model_path {MODEL_DIR}/{model} --features {features} --output_name {output_name}"

        # os.system(cmd)


        try:
            mod_summ = pd.read_csv(join(MODEL_DIR, model, "multivariate",f"{output_name}.csv"))
            # b, f = re.findall(r'\d+', model)
            mod_summ['AE'] = model
            mod_summ['features'] = features

            overall = pd.concat([overall, mod_summ])
        except FileNotFoundError:
            pass    
        overall.to_csv(MODEL_DIR + "AEvsRAD.csv", index=False)



if __name__ == "__main__":
    main()
    print(f"\nFinished.")
