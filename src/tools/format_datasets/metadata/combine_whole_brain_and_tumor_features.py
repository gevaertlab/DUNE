from os.path import join
import pandas as pd
import os
import argparse
import configparser


def ls_dir(path):
    dirs = [d for d in os.listdir(
        path) if os.path.isdir(os.path.join(path, d))]
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

    config['model_path'] = args.model_path
    config['tumor_path'] = join(
        args.model_path, "autoencoding/features/tumor.csv.gz")
    
    config['features_path'] = join(
        args.model_path, "autoencoding/features/features.csv.gz")

    return config


def main():

    config = parse_arguments()

    model_dir = config['model_path']
    radiomics = pd.read_csv(config['tumor_path'], index_col=0)
    features = pd.read_csv(config['features_path'], index_col=0)
    features.index.name = "eid"

    merged = pd.merge(features, radiomics, left_index=True, right_index=True)
    merged.columns = [i for i in range(merged.shape[1])]
    merged = merged.sort_index()

    merged.to_csv(join(model_dir, "autoencoding/features",
                  'whole_and_tumor.csv.gz'), index=True)


if __name__ == "__main__":
    main()
