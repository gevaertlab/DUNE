# %%
from radiomics import featureextractor
from os.path import join, isdir
import os
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm


DATASET = "TCGA"
MODALITIES = ["t1Gd","flair", "BoostBinary"]

ROOT = "/home/tbarba/projects/MultiModalBrainSurvival"
DATA = f"{ROOT}/data/MR/{DATASET}/images/"
OUTPUT = f"{ROOT}/data/MR/{DATASET}/metadata"

def ls_dir(path):
   dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
   return dirs


cases = ls_dir(DATA)
extractor = featureextractor.RadiomicsFeatureExtractor(f"{ROOT}/src/tools/format_datasets/pyradiomics.yaml",preCrop=True)
case_features = {}

for case in tqdm(cases):
    casedir = join(DATA, case)
    T1 = [join(casedir,f) for f in os.listdir(casedir) if f"{MODALITIES[0]}.nii" in f][0]
    flair = [join(casedir,f) for f in os.listdir(casedir) if f"{MODALITIES[1]}.nii" in f][0]
    segm = [join(casedir,f) for f in os.listdir(casedir) if f"{MODALITIES[2]}.nii" in f][0]

    T1_features = extractor.execute(T1, segm, label=1)
    flair_features = extractor.execute(flair, segm, label=1)

    T1_features = {f"T1_{k}":float(v) for k, v in T1_features.items() if "diagnostics" not in k}
    flair_features = {f"FLAIR_{k}":float(v) for k, v in flair_features.items() if "diagnostics" not in k}

    concat_features = T1_features | flair_features

    case_features[case] = [v for v in concat_features.values()]



output_file = join(OUTPUT, "0-pyradiomics.csv.gz")
df = pd.DataFrame.from_dict(case_features, orient="index")
df.index.name = "eid"

norm = pd.DataFrame(normalize(df), index=df.index)
norm.to_csv(output_file, index=True)

