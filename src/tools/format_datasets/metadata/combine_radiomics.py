from os.path import join
import pandas as pd
import os
from itertools import product
from tqdm import tqdm

os.chdir("/home/tbarba/projects/MultiModalBrainSurvival/")


DATASETS = ["UCSF", "TCGA", "UPENN"]
DATASETS = ["UCSF"]
# SEGM = ["", "_segm"]
SEGM = ["_segm2"]


def fuse(model):

    dataset, segm = model
    model_dir = f"outputs/UNet/finetuning/6b_4f_{dataset}{segm}"

    radiomics = f"data/MR/{dataset}/metadata/0-pyradiomics.csv.gz"
    radiomics = pd.read_csv(radiomics, index_col="eid")

    features = f"{model_dir}/autoencoding/features"
    features = join(features, "features.csv.gz")
    features = pd.read_csv(features, index_col=0)
    features.index.name = "eid"

    merged = pd.merge(features, radiomics, left_index=True, right_index=True)
    merged.columns = [i for i in range(merged.shape[1])]
    merged = merged.sort_index()

    merged.to_csv(join(model_dir, "autoencoding/features",
                  'features_and_radiomics.csv.gz'), index=True)


if __name__ == "__main__":

    models = [m for m in product(DATASETS, SEGM)]

    for model in tqdm(models):
        print(model)
        fuse(model)
