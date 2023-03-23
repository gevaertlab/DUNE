from os.path import join
import pandas as pd
import os

os.chdir("/home/tbarba/projects/MultiModalBrainSurvival/")



DATASET = "TCGA"
SEGM = "_segm"



if __name__ == "__main__":
    MODEL_DIR = f"outputs/UNet/finetuning/6b_4f_{DATASET}{SEGM}"
    RADIOMICS = f"data/MR/{DATASET}/metadata/0-pyradiomics.csv.gz"
    FEATURES = f"{MODEL_DIR}/autoencoding/features"


    radiomics = pd.read_csv(RADIOMICS, index_col="eid")

    features = join(FEATURES, "features.csv.gz")
    features = pd.read_csv(features, index_col=0)
    features.index.name = "eid"
    


    features.to_csv(join(MODEL_DIR, "autoencoding/features",'features.csv.gz'), index=True)


    merged = pd.merge(features, radiomics, left_index=True, right_index=True)

    merged.columns = range(merged.shape[1])

    merged = merged.sort_index()


    merged.to_csv(join(MODEL_DIR, "autoencoding/features",'features_and_radiomics.csv.gz'), index=True)