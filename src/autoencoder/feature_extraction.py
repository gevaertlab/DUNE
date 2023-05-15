import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from utils_ae import *



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


def main():
    config = parse_arguments("ae")
    model_path = config['model_path']

    # Restoring model
    net, _ = import_model(**config)
    net = nn.DataParallel(net)
    net = net.to(device)
    net.load_state_dict(torch.load(f"{model_path}/exports/best_model.pt"))

    # Restoring dataset
    fullLoader = torch.load(f"{model_path}/exports/fullLoader.pth")

    # Extracting features
    features = extract_features(net, fullLoader, device)
    features = pd.DataFrame.from_dict(features, orient="index")
    features = features.sort_index()
    features.index.name = "eid"
    features.to_csv(f"{model_path}/exports/features/whole_brain.csv.gz", index=True)

if __name__ == "__main__":
    main()
