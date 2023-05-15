import argparse
import os
import torch
import pandas as pd

from models.oldmodels import OldAE
from models.classicalAE import BaseAE
from models.resnetAE import RNet
from models.VAEs import U_VAE, VAE3D

import torch.nn as nn
from tqdm import tqdm
from utils import parse_arguments



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_model(type_ae, modalities, features, num_blocks, min_dims, num_epochs, model_path, **kwargs):
    
    model_path = os.path.join(model_path, "autoencoding/exported_data/best_model.pt")


    net, beta_dict = None, None
    # Initialize a model of our autoEncoder class on the device
    if type_ae.lower() in ["ae", "unet"]:
        net = BaseAE(len(modalities), features, num_blocks, type_ae=type_ae)
    elif type_ae.lower() in ["oldae", "oldaeu"]:
        net = OldAE(len(modalities), features, num_blocks, type_ae=type_ae)
    elif type_ae.lower() == "u_vae":
        net = U_VAE(len(modalities),  features, num_blocks, min_dims, hidden_size=2048)
    elif type_ae.lower() in ["vae3d"]:
        net = VAE3D(min_dims=min_dims)
    elif type_ae.lower() == "rnet":
        net = RNet(len(modalities))

    else:
        print("AE type should be one of followings : AE, UNet, VAE, RNet")

    net = nn.DataParallel(net)
    net = net.to(device)
    net.load_state_dict(torch.load(model_path))

    return net, beta_dict


def extract_features(net, val_dataloader, device):

    net.eval()
    results = {}

    for b_idx, (imgs, cases) in enumerate(tqdm(val_dataloader, colour="blue")):
        imgs = imgs.to(device)
        with torch.no_grad():
            _, extracted_features, _ = net(imgs)
            extracted_features = extracted_features.detach().cpu()

        for i, case in enumerate(cases):
            case_features = extracted_features[i,:, :, :, :].reshape(-1).numpy()

            results[case] = case_features

    return results


def main():
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    config = parse_arguments()
    # config = {k:v for k, v in config.items() if k in ["model_path","type_ae","modalities","features","num_blocks","min_dims"]}


    net,_ = import_model(**config)

    print("\nLoading datasets...")
    trainLoader = torch.load(f"{config['model_path']}/autoencoding/exported_data/trainLoader.pth")
    testLoader = torch.load(f"{config['model_path']}/autoencoding/exported_data/testLoader.pth")
    dict_loaders = {"train": trainLoader, "test": testLoader}

    final_df = pd.DataFrame()
    # feature extraction
    os.makedirs(f"{config['model_path']}/autoencoding/features", exist_ok=True)
    for dataset in ["train", "test"]:
        print(f"Extracting features for dataset : {dataset}")
        results = extract_features(
            net, dict_loaders[dataset], device)

        feature_dict = pd.DataFrame.from_dict(results, orient="index")

        final_df = pd.concat([final_df, feature_dict], axis=0)
        final_df.index.name = "eid"
        final_df = final_df.sort_index()
        final_df.to_csv(f"{config['model_path']}/autoencoding/features/features.csv.gz", index=True)

if __name__ == "__main__":
    main()
