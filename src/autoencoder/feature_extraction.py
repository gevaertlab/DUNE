import argparse
import os
import torch
import pandas as pd
from models import VAE
import torch.nn as nn
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help='model name')
    parser.add_argument("-d", "--model_dir", type=str, default = "outputs/UNet", help='model path')
    parser.add_argument("-n", "--num_mod", type=int, default = 2, help='number of modalities')
    parser.add_argument("-f", "--init_feat", type=int, default = 4, help='number of initial features')
    parser.add_argument("-b", "--num_blocks", type=int, default = 6, help='number of conv blocks')
    parser.add_argument("--unet", type=int, default = 1, help='unet architecture')

    args = parser.parse_args()
    args.model_dir = os.path.join(args.model_dir, args.model_name)
    args.unet = bool(args.unet)
    return args


def import_model(model_dir, num_mod, init_feat, num_blocks,unet, device):

    model_path = os.path.join(model_dir, "autoencoding/exported_data/best_model.pt")

    model = VAE(in_channels=num_mod, out_channels=num_mod,
                   init_features=init_feat, num_blocks=num_blocks, unet=unet)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    return model


def extract_features(net, val_dataloader, device):

    net.eval()
    results = {}

    for b_idx, (imgs, cases) in enumerate(tqdm(val_dataloader, colour="blue")):
        imgs = imgs.to(device)
        with torch.no_grad():
            _, extracted_features = net(imgs)
            extracted_features = extracted_features.detach().cpu()

        for i, case in enumerate(cases):
            case_features = extracted_features[i,:, :, :, :].reshape(-1).numpy()

            results[case] = case_features

    return results


def main():
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = import_model(args.model_dir, args.num_mod, args.init_feat, args.num_blocks, args.unet, device)

    print("\nLoading datasets...")
    trainLoader = torch.load(f'{args.model_dir}/autoencoding/exported_data/trainLoader.pth')
    testLoader = torch.load(f'{args.model_dir}/autoencoding/exported_data/testLoader.pth')
    dict_loaders = {"train": trainLoader, "test": testLoader}

    final_df = pd.DataFrame()
    # feature extraction
    os.makedirs(f"{args.model_dir}/autoencoding/features", exist_ok=True)
    for dataset in ["train", "test"]:
        print(f"Extracting features for dataset : {dataset}")
        results = extract_features(
            net, dict_loaders[dataset], device)

        feature_dict = pd.DataFrame.from_dict(results, orient="index")
        # feature_dict.to_csv(
            # f"{args.model_dir}/autoencoding/features/{dataset}_features.csv.gz", index=True)

        final_df = pd.concat([final_df, feature_dict], axis=0)
        final_df.index.name = "eid"
        final_df = final_df.sort_index()
        final_df.to_csv(f"{args.model_dir}/autoencoding/features/features.csv.gz", index=True)

if __name__ == "__main__":
    main()
