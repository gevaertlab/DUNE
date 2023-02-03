import os
import pandas as pd
from os.path import join
import re

MODEL_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/pretraining"

def ls_dironly(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs
    


def univariate_models(list_of_models):

    for _, model in enumerate(list_of_models):
        print("Evaluating ", model)
        cmd = f"Rscript aspects/0_MRI/predictions/univariate.r \
		--config {MODEL_DIR}/{model}/config/univariate.json" 

        os.system(cmd)

def main():
    ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOT_DIR)

    list_of_models = [
        "UNet_6b_4f_UKfull",
        "UNet_6b_8f_UKfull",
        "UNet_5b_4f_UKfull",
        "UNet_5b_8f_UKfull"
        ]

    # univariate_models(list_of_models)

    overall = pd.DataFrame()
    for model in list_of_models:
        try:
            mod_summ = pd.read_csv(join(MODEL_DIR, model, "univariate","0-univariate_results.csv"))
            b, f = re.findall(r'\d+', model)
            mod_summ['model'] = f"{b}B_{f}F"

            overall = pd.concat([overall, mod_summ])
        
        except FileNotFoundError:
            pass

    overall.to_csv(MODEL_DIR + "/univ_summaryNORM.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"\nFinished.")
