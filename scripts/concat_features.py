
# %%
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metadata", type=str, required=True,
                        help='input metadata')
    parser.add_argument("-f", "--features_dir", type=str, required=True,
                        help='input features data')


    args = parser.parse_args()

    return args

# %%
def main():

    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival/")
    args = parse_arguments()

    features_files = [
        f for f in os.listdir(args.features_dir) if f.endswith("csv") if "concat" not in f
        ]
    
    metadata_path = args.metadata
    metadata = pd.read_csv(metadata_path)


    if "eid" in metadata.columns:
        metadata = metadata.drop_duplicates().set_index("eid")
    elif "disease" in metadata.columns:
        metadata = metadata[["case", "age","sex","survival","disease","grade"]].set_index("case")
    else:
        metadata = metadata[["case","case_id","survival_months","grade","vital_status", "survival_bin","grade_binary", "IDH1","IDH1_bin"]].drop_duplicates().set_index("case")

    final = pd.DataFrame()
    for dataset in features_files:

        features_path = os.path.join(args.features_dir, dataset)
        features = pd.read_csv(features_path)
        features["case"] = features["Unnamed: 0"]
        features = features.drop(["Unnamed: 0"], axis=1).set_index("case")

        merged = metadata.merge(features, how="inner", left_index=True, right_index= True)
        merged.index = merged.index.rename("case")

        final = pd.concat([final, merged])

    final = shuffle(final)
    print("Exporting csv...")
    
    train, val, test = np.split(
        final, [int(.6*len(final)), int(.8*len(final))])

    train.to_csv(args.features_dir + '/concat_train.csv.gz', index=True)
    val.to_csv(args.features_dir + '/concat_val.csv.gz', index=True)
    test.to_csv(args.features_dir + '/concat_test.csv.gz', index=True)
    
    print("Finished.")

if __name__ == "__main__":
    main()


# %%
