import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from argparse import ArgumentParser
from os.path import join
import configparser
import os


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_file',
                        type=str,
                        help='config file path')

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

    config = {**model, **data, **conf, **predictions}

    return config


def create_embeddings(features_path, meta_path, variables):
    data = pd.read_csv(features_path, index_col="eid").sort_index()
    metadata = pd.read_csv(
        meta_path, index_col="eid").loc[data.index].sort_index()

    variables = list(pd.read_csv(variables).query("keep_model")["var"])
    try:
        variables.remove("death")
    except:
        pass

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    return embedding, metadata, variables


def plot_graph(var, embedding, metadata):
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1], s=1,
        c=metadata[var]
    )

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
        features = join(mod, "exports/features/whole_brain.csv.gz")
    meta = join(config["data_path"], config["dataset"],
                "metadata", config["metadata"])
    variables = join(config["data_path"], config["dataset"],
                     "metadata", config["variables"])

    embedding, metadata, variables = create_embeddings(
        features, meta, variables)

    output_file = join(mod, "umap", mod.split("/")[-1]+".pdf")
    os.makedirs(join(mod, "umap"), exist_ok=True)

    pp = PdfPages(output_file)
    for var in tqdm(variables, colour="blue"):
        pp.savefig(plot_graph(var, embedding, metadata))
    pp.close()
