# %%
from radiomics.featureextractor import RadiomicsFeatureExtractor
from os.path import join, isdir
from os import listdir
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm
from multiprocessing import Pool
from collections import ChainMap
import sys

DATASET = sys.argv[1]

DATA = f"data/MR/{DATASET}/processed"
OUTPUT = f"data/MR/{DATASET}/metadata"
MODALITIES = [m.lower() for m in ["normT1_crop", "reconstructed", "brain_mask"]]
OUTPUT_FILE = join(OUTPUT, "0-WB_pyradiomicsREC.csv.gz")


def ls_dir(path):
    dirs = [d for d in listdir(path) if isdir(join(path, d))]
    return dirs


def extract_radiomics(case):

    casedir = join(DATA, case)
    segm = [join(casedir, f) for f in listdir(casedir)
            if f"{MODALITIES[2]}.nii" in f.lower()][0]

    T1 = [join(casedir, f) for f in listdir(casedir)
          if f"{MODALITIES[0]}.nii" in f.lower()][0]
    T1_features = extractor.execute(T1, segm, label=1)

    try:
        flair = [join(casedir, f) for f in listdir(casedir)
                 if f"{MODALITIES[1]}.nii" in f.lower()][0]
        flair_features = extractor.execute(flair, segm, label=1)
    except:
        flair_features = {}
        pass

    T1_features = {f"T1_{k}": float(
        v) for k, v in T1_features.items() if "diagnostics" not in k}
    flair_features = {f"FLAIR_{k}": float(
        v) for k, v in flair_features.items() if "diagnostics" not in k}

    concat_features = T1_features | flair_features

    res = {case: [v for v in concat_features.values()]}

    return res


if __name__ == "__main__":

    extractor = RadiomicsFeatureExtractor(
        f"src/tools/format_datasets/metadata/pyradiomics.yaml", preCrop=True)

    cases = ls_dir(DATA)

    with Pool(10) as p:
        case_features = list(
            tqdm(p.imap(extract_radiomics, cases), colour="yellow", total=len(cases)))

    case_features = dict(ChainMap(*case_features))
    df = pd.DataFrame.from_dict(case_features, orient="index")
    df.index.name = "eid"

    norm = pd.DataFrame(normalize(df), index=df.index)
    norm.to_csv(OUTPUT_FILE, index=True)


# %%
