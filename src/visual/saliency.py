# %%
import argparse
import os
import torch
import pandas as pd
from models import UNet3D, CompleteRegressor
from torchvision import transforms as transforms
from datasets import BrainImages
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import numpy as np
from os.path import join
from matplotlib import pyplot as plt

import torch.nn as nn
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help='model name')
    parser.add_argument("-d", "--model_dir", type=str, default = "outputs/UNet", help='model path')
    parser.add_argument("-n", "--num_mod", type=int, default = 2, help='number of modalities')
    parser.add_argument("-f", "--init_feat", type=int, default = 8, help='number of initial features')
    parser.add_argument("-b", "--num_blocks", type=int, default = 6, help='number of conv blocks')

    args = parser.parse_args()
    args.model_dir = os.path.join(args.model_dir, args.model_name)

    return args


def import_model(model_dir, num_mod, init_feat, num_blocks, device):

    model_path = os.path.join(model_dir, "autoencoding/exported_data/best_model.pt")

    model = UNet3D(in_channels=num_mod, out_channels=num_mod,
                   init_features=init_feat, num_blocks=num_blocks)
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


def train_loop(model, dataloader, optimizer, device, train):

    loss_list = []
    col = "magenta" if train else "cyan"
    model.train() if train else model.eval()

    for idx, (img, label) in enumerate(tqdm(dataloader, colour=col)):
            
        with torch.set_grad_enabled(train):
            img = img.to(device)
            label = label.to(device)
            loss = model.compute_loss(img, label)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        if idx == 10:
            break

    val_loss = np.mean(loss_list)

    return val_loss





if __name__=='__main__':
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    # args = parse_arguments()


    #######################################################
    ROOT_DATA = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR"
    dataset = "UCSF"
    model_dir = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/6b_4f_UCSF_segm"
    min_dims = [  158,   198,   153]
    num_mod = 3
    init_feat = 4
    num_blocks = 6
    modalities = ["T1c","FLAIR","tumor_segmentation"]
    data_path = join(ROOT_DATA, dataset,"images")
    metadata = join(ROOT_DATA, dataset, "metadata", "0-UCSF_metadata_encoded.csv")
    variable = "age_at_mri"
    task = "regression"
    batch_size=5
    ##########################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    encoder = import_model(model_dir , num_mod, init_feat, num_blocks, device)
    # last_layer = torch.load(
    #     "/home/tbarba/projects/MultiModalBrainSurvival/src/visual/test.path")


    
    # net = CompleteRegressor(
    #     encoder=encoder,
    #     loss_fn=torch.nn.MSELoss(),
    #     n_inputs=3072,
    #     l1_lambda=0.00,
    #     l2_lambda=0.05)
    
    # net_dict = net.state_dict()
    # for l in last_layer.keys():
    #     net_dict[l] = last_layer[l]
    # net.load_state_dict(net_dict)

    net = encoder

    net = net.to(device)
    # Freeze the encoder
    for name, p in net.named_parameters():
        if "encoder.module" in name:
            p.requires_grad = False


    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)


    # Data
    print("\nLoading datasets...")
    metadata = pd.read_csv(metadata, index_col="eid")
    metadata = metadata[variable]


    normalTransform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])
    totalData = BrainImages(dataset, data_path, modalities,
                        min_dims, metadata, transforms=normalTransform)
    
    totalLoader = DataLoader(
        totalData, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

# %%
    for img, lab in totalLoader:

        img.requires_grad = True
        out = net(img)

        out[0,0].backward()

        grads = img.grad.detach().numpy()

        img_s = img.detach().numpy()

        plt.imshow(img_s[0,1, 70,:,:], cmap="Greys_r")
        plt.imshow(grads[0,1, 70,:,:], alpha=0.4, cmap="Reds_r")

        break


# %%
