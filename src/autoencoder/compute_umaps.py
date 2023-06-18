import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from argparse import ArgumentParser
from os.path import join
import configparser
import os
import seaborn as sns


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_file',
                        type=str,
                        help='config file path')

    parser.add_argument('-k', '--keep_single', type=str, default="False",
                    help='for feature extraction - keep single modalities', required=False)

    args = parser.parse_args()

    config_file = join(
        "outputs", args.config_file, "config.cfg")
    conf_parser = configparser.ConfigParser()
    conf_parser.read(config_file)

    conf = {k: eval(v) for k, v in dict(conf_parser["config"]).items()}
    model = {k: eval(v) for k, v in dict(conf_parser["model"]).items()}
    data = {k: eval(v) for k, v in dict(conf_parser["data"]).items()}
    predictions = {k: eval(v) for k, v in dict(
        conf_parser["predictions"]).items()}

    model["model_path"] = join(model["model_path"], model['model_name'])

    conf["keep_single"] = eval(args.keep_single)

    config = {**model, **data, **conf, **predictions}

    return config


def create_embeddings(features_path, meta_path, variables):
    data = pd.read_csv(features_path, dtype={"eid": str})
    data = data.set_index("eid").sort_index()
    metadata = pd.read_csv(
        meta_path,  dtype={"eid": str}).set_index("eid")
    metadata = metadata.loc[data.index].sort_index()

    variables = list(pd.read_csv(variables).query("keep_model")["var"])
    try:
        variables.remove("death")
    except:
        pass

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    embedding = pd.DataFrame(embedding, index=data.index)
    embedding = embedding.merge(metadata, left_index=True, right_index=True)

    return embedding, variables


def plot_graph(embedding, var):

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(ax=ax, data=embedding,
                    x=0, y=1, hue=var, s=8, edgecolor="black", alpha=0.6)
    ax.set_title(var)

    plt.close()

    return fig


if __name__ == "__main__":
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")

    config = parse_arguments()
    mod = config["model_path"]

    if config["features"] == "radiomics":
        features = join(config["data_path"], config["dataset"],
                        "metadata", config["pyradiomics"])
    else:
        wb_feats = "whole_brain" if not config["keep_single"] else "wb_per_mod"
        features = join(mod, f"exports/features/{wb_feats}.csv.gz")

    if type(config["dataset"]) == list:
        meta = config["metadata"]
        variables = config["variables"]
    else:
        meta = join(config["data_path"], config["dataset"],
                    "metadata", config["metadata"])
        variables = join(config["data_path"], config["dataset"],
                         "metadata", config["variables"])

    # meta = join(config["data_path"], config["dataset"],
    #             "metadata", config["metadata"])

    embedding, variables = create_embeddings(
        features, meta, variables)

    output_file = join(mod, "umap", mod.split("/")[-1]+".pdf")
    os.makedirs(join(mod, "umap"), exist_ok=True)

    pp = PdfPages(output_file)
    for var in tqdm(variables, colour="blue"):
        pp.savefig(plot_graph(embedding, var))
    pp.close()
