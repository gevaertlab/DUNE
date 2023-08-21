import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
import os
import configparser
from os.path import join
from argparse import ArgumentParser
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import umap
import pandas as pd


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_file',
                        type=str,
                        help='config file path')

    parser.add_argument('-k', '--keep_single', type=str, default="False",
                        help='for feature extraction - keep single modalities', required=False)

    parser.add_argument('-f', '--features', type=str,
                        help='features files', required=True)

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
    config["features"] = args.features
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


def plot_graph(embedding, var, pal):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 6))

    g = sns.scatterplot(ax=ax, data=embedding,
                        x=0, y=1, hue=var, s=9,
                        edgecolor="black",
                        linewidth=0.05,
                        palette=pal,
                        alpha=1)
    # sns.move_legend(g, "center right", bbox_to_anchor=(.55, .45))
    g._legend.remove()
    ax.set_title(var)

    plt.close()

    return fig


if __name__ == "__main__":
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")

    config = parse_arguments()
    mod = config["model_path"]

    features = join(mod, f"exports/features/{config['features']}.csv.gz")

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

    output_file = join(mod, "multivariate", config["features"] + "_UMAP.pdf")
    os.makedirs(join(mod, "multivariate"), exist_ok=True)

    to_plot = {
        "Volume_of_grey_matter": "plasma",
        "Sexe": "Spectral",
        "Alcohol_intake_frequency": "plasma",
        "Scanner_transverse_Y_brain_position": "plasma",
        "UK_Biobank_assessment_centre": "Spectral",
        # "Volumetric_scaling_from_T1_head_image_to_standard_space":"plasma",
    }

    pp = PdfPages(output_file)
    embedding.to_csv(output_file + ".csv.gz", index=True)

    fig, ax = plt.subplots(len(to_plot), 1)
    for idx, var in tqdm(enumerate(to_plot), desc=config['features']):
        print(var)
        g=sns.scatterplot(data=embedding,
                            x=0, y=1, hue=var, s=9,
                            edgecolor="black",
                            linewidth=0.05,
                            palette=to_plot[var],
                            alpha=1,
                            ax=ax[idx])
        g._legend.remove()
        g.set_title(var)
        # p = plot_graph(embedding, var, to_plot[var])
        # pp.savefig(p)
    plt.savefig(f"pile_{output_file}.pdf")
    pp.close()
