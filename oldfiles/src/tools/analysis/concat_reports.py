import os
from os.path import join
import pandas as pd


MODELS = "data/outputs"

FROM = "fused"



dfs = {f:pd.read_csv(join(FROM,f), index_col=0) for f in os.listdir(FROM) if f.endswith("csv")}

synth = pd.DataFrame()
for f, df in dfs.items():
    dataset = f.split("_")[0]
    feat = f.split("_")[1]
    df["feat"]= feat[:-4]
    df["dataset"]= dataset
    synth = pd.concat([synth, df], axis=0)

    synth.to_csv(join(MODELS,"tumor_crop.csv"), index=True)