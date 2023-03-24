# %%
from radiomics import featureextractor
from os.path import join, isdir
import os
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm
import argparse
import configparser
from os.path import join



def ls_dir(path):
   dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
   return dirs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        help='model_path')
    
    args = parser.parse_args()

    config_file = join(args.model_path, "config/ae.cfg")
    config = configparser.ConfigParser()
    config.read(config_file)
    config = dict(config["config"])
    
    config['modalities'] = eval(config['modalities'] )
    config['data'] = join(config['data_path'], "images") 
    config['output'] = join(config['data_path'], "metadata") 

    return config



if __name__ == "__main__":
    
    config = parse_arguments()


    extractor = featureextractor.RadiomicsFeatureExtractor(f"/src/tools/format_datasets/metadata/pyradiomics.yaml",preCrop=True)
    
    DATA = config['data'] 
    OUTPUT = config['output']
    MODALITIES = [m.lower() for m in config['modalities'] ]

    cases = ls_dir(DATA)
    case_features = {}
    for case in tqdm(cases):
        casedir = join(DATA, case)
        T1 = [join(casedir,f) for f in os.listdir(casedir) if f"{MODALITIES[0]}.nii" in f.lower()][0]
        flair = [join(casedir,f) for f in os.listdir(casedir) if f"{MODALITIES[1]}.nii" in f.lower()][0]
        segm = [join(casedir,f) for f in os.listdir(casedir) if f"{MODALITIES[2]}.nii" in f.lower()][0]

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

