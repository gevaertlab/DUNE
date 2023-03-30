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
        ["UNet", "6b_4f_UCSF_segm", "features", "UCSF_features"],
        ["UNet", "6b_4f_UCSF_segm", "radiomics", "UCSF_radiomics"],
        ["UNet", "6b_4f_UCSF_segm", "combined", "UCSF_combined"],

        ["VAE", "VAE_UCSF_segm", "features", "UCSF_features"],
        ["VAE", "VAE_UCSF_segm", "combined", "UCSF_combined"],

        ["UNet", "6b_4f_TCGA_segm","features", "TCGA_features"],
        ["UNet", "6b_4f_TCGA_segm","radiomics", "TCGA_radiomics"],
        ["UNet", "6b_4f_TCGA_segm","combined", "TCGA_combined"],

        ["VAE", "VAE_TCGA_segm","features", "TCGA_features"],
        ["VAE", "VAE_TCGA_segm","combined", "TCGA_combined"],

        ["UNet", "6b_4f_UPENN_segm", "features", "UPENN_features"],
        ["UNet", "6b_4f_UPENN_segm", "radiomics", "UPENN_radiomics"],
        ["UNet", "6b_4f_UPENN_segm", "combined", "UPENN_combined"],

        ["VAE", "VAE_UPENN_segm", "features", "UPENN_features"],
        ["VAE", "VAE_UPENN_segm", "combined", "UPENN_combined"]
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
        overall.to_csv(MODEL_DIR + "AEvsRAD3.csv", index=False)



if __name__ == "__main__":
    main()
    print(f"\nFinished.")
