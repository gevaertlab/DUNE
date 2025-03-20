# %% init
import pandas as pd
from pathlib import Path
import numpy as np

HOME = Path("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/")

datasets = ["UKBIOBANK", "UCSF", "UPENN", "TCGA", "SCHIZO", "ADNI"]


metadata = {
    "UKB": "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/metadata/0-UKB_metadata_encoded.csv.gz",
    "UCSF": "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UCSF/metadata/0-UCSF_metadata_encoded.csv",
    "UPENN": "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/0-UPENN_metadata_encoded.csv",
    "TCGA": "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-TCGA_metadata_encoded.csv",
    "SCHIZO": "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/SCHIZO/metadata/0-SCHIZO_metadata_encoded.csv",
}


metadata = {d: pd.read_csv(p, index_col="eid") for d, p in metadata.items()}


final_df = []
for d in metadata.keys():
    df = metadata[d]
    df["dataset"] = d
    df.rename(
        mapper={"Sexe": "Sex", "sex": "Sex", "Gender": "Sex"}, axis=1, inplace=True)

    df["Cancer"] = np.where(df["dataset"].isin([
                                  "UCSF", "UPENN", "TCGA"]), True, False).astype(int)
    df = df[["dataset", "Sex", "Cancer"]]
    final_df.append(df)


final_df = pd.concat(final_df)


final_df.to_csv("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/all_cohorts.csv.gz", index=True)
# %%
