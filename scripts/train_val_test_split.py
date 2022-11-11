# %%
import os
import pandas as pd
import numpy as np
from alive_progress import alive_it

WD = "/home/tbarba/projects/MultiModalBrainSurvival/"
FOLDER = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/UNet3D_TCGA_Feat4/results/features"


def main():
    os.chdir(WD)

    print("Gathering filenames")
    subset_filelist = [f for f in alive_it(os.listdir(
        PATCHES_PATH)) if os.path.isdir(PATCHES_PATH + "/" + f)]
    survival_data = pd.read_csv(
        "survival_data/ffpe_fulldata.csv", index_col=0).set_index("wsi_file_name")

    # Filtering survival dataframe
    survival_data = survival_data.loc[[f.split('.')[0] for f in alive_it(
        subset_filelist) if f.split('.')[0] in survival_data.index]].reset_index()

    survival_data["wsi_file_name"] = [f for f in alive_it(subset_filelist) if f.split('.')[
        0] in survival_data.index]

    # split into train val and test sets
    train, val, test = np.split(
        survival_data, [int(.6*len(survival_data)), int(.80*len(survival_data))])

    train.to_csv(SURVIVAL_DATA_PATH + '/train.csv')
    val.to_csv(SURVIVAL_DATA_PATH + '/val.csv')
    test.to_csv(SURVIVAL_DATA_PATH + '/test.csv')


if __name__ == "__main__":
    main()
