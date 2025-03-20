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

    config_file = join(args.model_path, "config.cfg")
    conf_parser = configparser.ConfigParser()
    conf_parser.read(config_file)
    # conf_parser = dict(config["config"])


    conf = {k: eval(v) for k, v in dict(conf_parser["config"]).items()}
    model = {k: eval(v) for k, v in dict(conf_parser["model"]).items()}
    data = {k: eval(v) for k, v in dict(conf_parser["data"]).items()}
    predictions = {k: eval(v) for k, v in dict(conf_parser["predictions"]).items()}

    config = {**conf, **model, **data, **predictions}


    config['model_path'] = args.model_path
    config['radiomics_path'] = join(
        config['data_path'], config['dataset'], "metadata/0-WB_pyradiomics.csv.gz")
    
    config['features_path'] = join(
        args.model_path, "exports/features/whole_brain.csv.gz")

    return config


def main():

    config = parse_arguments()

    model_dir = config['model_path']
    radiomics = pd.read_csv(config['radiomics_path'], index_col="eid")
    features = pd.read_csv(config['features_path'], index_col=0)
    features.index.name = "eid"

    merged = pd.merge(features, radiomics, left_index=True, right_index=True)
    merged.columns = [i for i in range(merged.shape[1])]
    merged = merged.sort_index()

    merged.to_csv(join(model_dir, "exports/features",
                  'wb_and_radiomics.csv.gz'), index=True)


if __name__ == "__main__":
    main()
