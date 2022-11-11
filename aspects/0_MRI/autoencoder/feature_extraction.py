import argparse
import os
import torch
import numpy as np
import pandas as pd
from models import UNet3D
import torch.nn as nn
import torchvision.transforms as transforms


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help='model name')
    parser.add_argument("-d", "--model_dir", type=str, default = "outputs/UNet", help='model path')
    parser.add_argument("-n", "--num_mod", type=int, default = 2, help='number of modalities')

    args = parser.parse_args()

    args.model_dir = os.path.join(args.model_dir, args.model_name)

    return args


def import_model(model_dir, num_mod, device):

    # model_dir = f"outputs/training/{model_name}"
    model_path = os.path.join(model_dir, "exported_data/model.pth")

    model = UNet3D(in_channels=num_mod, out_channels=num_mod, init_features=4)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    return model


def extract_features(net, val_dataloader, device):

    net.eval()
    results = {}


    for b_idx, (imgs, cases) in enumerate(val_dataloader):
        imgs = imgs.to(device)
        with torch.no_grad():
            _, extracted_features = net(imgs)
            extracted_features = extracted_features.detach().cpu()

        for i, case in enumerate(cases):
            # try:
            case_features = extracted_features[i,
                                                :, :, :, :].reshape(-1).numpy()


            results[case] = case_features

    return results


def main():
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = import_model(args.model_dir, args.num_mod, device)

    trainLoader = torch.load(f'{args.model_dir}/exported_data/trainLoader.pth')
    testLoader = torch.load(f'{args.model_dir}/exported_data/testLoader.pth')
    dict_loaders = {"train": trainLoader, "test": testLoader}

    # feature extraction

    os.makedirs(f"{args.model_dir}/results/features", exist_ok=True)
    for dataset in ["train", "test"]:
        print(f"extracting features for dataset : {dataset}")
        results = extract_features(
            net, dict_loaders[dataset], device)


        feature_dict = pd.DataFrame.from_dict(results, orient="index")
        feature_dict.to_csv(
            f"{args.model_dir}/results/features/{dataset}_features.csv", index=True)



if __name__ == "__main__":
    main()
