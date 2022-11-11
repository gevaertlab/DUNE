# %%
import os
import argparse
import pandas as pd
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--survival", type=str, required=True,
                        help='input survival data')
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
    
    surv_path = args.survival
    survival = pd.read_csv(surv_path)
    survival = survival[["case", "survival_months","vital_status"]].drop_duplicates().set_index("case")
    final = pd.DataFrame()
    for dataset in features_files:

        features_path = os.path.join(args.features_dir, dataset)
        features = pd.read_csv(features_path).set_index("Unnamed: 0")

        merged = survival.merge(features, how="inner", left_index=True, right_index= True)
        merged.index = merged.index.rename("case")

        final = pd.concat([final, merged])

    print("Exporting csv...")
    
    train, val, test = np.split(
        final, [int(.5*len(final)), int(.75*len(final))])

    train.to_csv(args.features_dir + '/concat_train.csv', index=True)
    val.to_csv(args.features_dir + '/concat_val.csv', index=True)
    test.to_csv(args.features_dir + '/concat_test.csv', index=True)


    # final.to_csv(os.path.join(args.features_dir, "fulldata_surv.csv"), index=True)
    print("Finished.")

if __name__ == "__main__":
    main()


# %%
