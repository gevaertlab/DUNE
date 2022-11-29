
# %%
import os
import argparse
import pandas as pd
import numpy as np


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

    # metadata = metadata.rename(columns = {
    #     "eid":"case","31-0.0":"sex", "53-0.0": "start_date", "21022-0.0":"age",
    #     "12144-2.0":"height", "25009-2.0":"norm_vol", "25010-2.0":"brain_vol" })

    metadata = metadata[["eid", "sex", "age", "sleep", "alcohol_freq", "alcohol_status", "smoking", "greymat_vol", "brain_vol", "norm_brainvol",
                         "fluidency", "digits_symbols", "depression", "height", "weight", "bmi", "diastole"]].drop_duplicates().set_index("eid")
    final = pd.DataFrame()
    for dataset in features_files:

        features_path = os.path.join(args.features_dir, dataset)
        features = pd.read_csv(features_path)
        features["case"] = features["Unnamed: 0"]
        features = features.drop(["Unnamed: 0"], axis=1).set_index("case")

        merged = metadata.merge(features, how="inner", left_index=True, right_index= True)
        merged.index = merged.index.rename("case")

        final = pd.concat([final, merged])

    print("Exporting csv...")
    
    train, val, test = np.split(
        final, [int(.6*len(final)), int(.8*len(final))])

    train.to_csv(args.features_dir + '/concat_train.csv', index=True)
    val.to_csv(args.features_dir + '/concat_val.csv', index=True)
    test.to_csv(args.features_dir + '/concat_test.csv', index=True)
    
    print("Finished.")

if __name__ == "__main__":
    main()


# %%
