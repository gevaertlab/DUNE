import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json

DATASET = "UCSF"

METADATA_PATH = f"/home/tbarba/projects/MultiModalBrainSurvival/data/MR/{DATASET}/metadata/0-{DATASET}_metadata_encoded.csv"



df = pd.read_csv(METADATA_PATH, index_col=0)
index = list(df.index)

train, test = train_test_split(df, test_size = 0.80)


cohorts = {
    "train":train,
    "test":test
}

pd.DataFrame().from_dict(cohorts, orient="columns")