# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from os.path import join
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models import ElasticLinear
from sklearn.metrics import r2_score, accuracy_score
from tqdm import tqdm

# %%

def create_fulldataset(csv_paths, metadata_path):

    metadata = pd.read_csv(metadata_path, index_col="eid")

    list_features = []
    for f in csv_paths:
        df = pd.read_csv(f)
        list_features.append(df)
    features = pd.concat(list_features, axis=0, ignore_index=True)
    features["eid"] = features["Unnamed: 0"]
    features = features.drop(["Unnamed: 0"], axis=1).set_index("eid")

    merged = metadata.merge(features, how="inner",
                            left_index=True, right_index=True)

    return merged


def create_train_test_datasets(merged, var, task):

    nrow_init = merged.shape[0]

    if task != "survival":
        labels = merged[var]
        labels = np.array(labels)

        merged = merged.loc[~np.isnan(labels)]
        labels = labels[~np.isnan(labels)]
    else:
        time = merged[f"{var}_delay"].astype(float)
        event = merged[f"{var}_event"].astype(bool)
        labels = np.array([(e, t) for e, t in zip(event, time)], dtype=[
                          ('Status', '?'), ('Survival_in_days', '<f8')])

        merged = merged.loc[~np.isnan(time)]
        labels = labels[~np.isnan(time)]

    features = merged[[k for k in merged.columns if k.isdigit()]]

    missing_rate = round(1-(features.shape[0] / nrow_init), 2)

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(features), np.array(labels), test_size=0.2, random_state=12)
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    return X_train, X_test, y_train, y_test, missing_rate
    

def create_dataloaders(X_train, X_test, y_train, y_test):

    dataset_train = TensorDataset(X_train, y_train)
    dataloader_train = DataLoader(
        dataset_train, batch_size=X_train.size(0), shuffle=True, num_workers=1)
    dataset_test = TensorDataset(X_test, y_test)

    dataloader_test = DataLoader(
        dataset_test, batch_size=X_test.size(0), shuffle=True, num_workers=1)

    num_feats = X_train.size(1)

    return dataloader_train, dataloader_test, num_feats





# %%


METADATA_PATH = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/0-UPENN_metadata_encoded.csv"
VARIABLE_LIST = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UPENN/metadata/0-variable_list.csv"
#######################################################
ROOT_DATA = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR"
dataset = "UPENN"
model_dir = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/6b_4f_UPENN_segm"
min_dims = [  158,   198,   153]
num_mod = 3
init_feat = 4
num_blocks = 6
modalities = ["T1c","FLAIR","tumor_segmentation"]
data_path = join(ROOT_DATA, dataset,"images")
metadata = join(ROOT_DATA, dataset, "metadata", "0-UPENN_metadata_encoded.csv")
batch_size=5
task="classif"
##########################################################



def train_loop(model, dataloader, optimizer, device, train):

    loss_list = []
    model.train() if train else model.eval()
    col = "magenta" if train else "cyan"

    for img, label in dataloader:
            
        with torch.set_grad_enabled(train):
            img = img.to(device)
            label = label.to(device)
            loss = model.compute_loss(img, label)


        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

    val_loss = np.mean(loss_list)

    return val_loss





if __name__=='__main__':
    csv_paths = [join(model_dir, "autoencoding/features",
                          f"{file}_features.csv.gz") for file in ["train", "test"]]
    merged = create_fulldataset(csv_paths, METADATA_PATH)

    var = "IDH1"
    classifier = True
    num_class=3


    

    X_train, X_test, y_train, y_test, missing_rate = create_train_test_datasets(
            merged, var, task)

    trainLoader, valLoader, num_feats= create_dataloaders(X_train, X_test, y_train, y_test)
    device = torch.device("cpu")
    
    loss_fun = torch.nn.CrossEntropyLoss() if classifier else torch.nn.MSELoss()
    scoring = accuracy_score if classifier else r2_score

    model = ElasticLinear(
        loss_fn=loss_fun,
        n_inputs=num_feats, num_class=num_class,
        l1_lambda=0.8,
        l2_lambda=0.1,
        classifier=classifier
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = [], []

    for epoch in tqdm(range(500)):
        epoch_train_loss = train_loop(model, trainLoader, optimizer, device, train=True)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = train_loop(model, valLoader, optimizer, device, train=False)
        val_losses.append(epoch_val_loss)

        print(model(X_test)[:5])

    import seaborn as sns
    sns.displot(model(X_test).detach().numpy())
    a, y_pred = torch.max(model(X_test), 1)
    score = scoring(y_pred.detach().numpy(), y_test)


    
# %%


import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.plot(val_losses)

# %%
