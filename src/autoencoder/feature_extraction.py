import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import utils_ae as ut


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(net, fullLoader, device):

    net.eval()
    results = {}

    for imgs, case in tqdm(fullLoader, colour="blue", desc="Features extraction"):
        case = case[0]
        imgs = imgs.to(device)
        with torch.no_grad():
            _, extracted_features, _ = net(imgs)

        extracted_features = extracted_features.detach().cpu().view(-1).numpy()

        results[case] = extracted_features

    return results


def rearrange_df(features):

    feats = [c for c in features.columns if str(c).isdigit()]
    features = features.reset_index()
    features["eid"] = features["eid"].astype(str)
    features[["eid", "mod"]] = features["eid"].str.split("__", expand=True)
    features = pd.pivot(features, index="eid", values=feats, columns="mod")
    features.columns = range(features.shape[1])
    return features


def main():
    config = ut.parse_arguments("ae")
    model_path = config['model_path']
    keep_single = config["keep_single"]

    # Restoring model
    net, _ = ut.import_model(**config, device=device)
    net = nn.DataParallel(net)
    net = net.to(device)
    if config["other_model"]:
        print("Using other model : ", config["other_model"])
        dict_path = f"{config['other_model']}/exports/best_model.pt"
    else:
        dict_path = f"{model_path}/exports/best_model.pt"
    net.load_state_dict(torch.load(dict_path))

    # Restoring dataset
    fullLoader = torch.load(f"{model_path}/exports/fullLoader.pth")

    # Extracting features
    features = extract_features(net, fullLoader, device)
    features = pd.DataFrame.from_dict(features, orient="index")
    features = features.sort_index()
    features.index.name = "eid"

    if config["single_mod"] and not keep_single:
        features = rearrange_df(features)

    output = "whole_brain" if not keep_single else "wb_per_mod"

    features.to_csv(
        f"{model_path}/exports/features/{output}.csv.gz", index=True)


if __name__ == "__main__":
    main()
