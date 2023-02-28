# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from os.path import join
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from models import ElasticLinear
from torchmetrics import R2Score

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

    return dataloader_train, dataloader_test, X_train.size(1)


# %%
# MODELS

def plot_convergence(train_loss):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(train_loss)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    fig.show()




# %%


METADATA_PATH = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UCSF/metadata/0-UCSF_metadata_encoded.csv"
VARIABLE_LIST = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UCSF/metadata/0-variable_list.csv"
MODEL_PATH = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/finetuning/6b_4f_UCSF_segm"

OUTPUT_DIR = "test"




if __name__=='__main__':
    csv_paths = [join(MODEL_PATH, "autoencoding/features",
                          f"{file}_features.csv.gz") for file in ["train", "test"]]
    merged = create_fulldataset(csv_paths, METADATA_PATH)

    var = "age_at_mri"
    task = "regression"

    X_train, X_test, y_train, y_test, missing_rate = create_train_test_datasets(
            merged, var, task)

    dataloader_train, dataloader_test, trainsize= create_dataloaders(X_train, X_test, y_train, y_test)


    model = ElasticLinear(
        loss_fn=torch.nn.MSELoss(),
        n_inputs=trainsize,
        l1_lambda=0.00,
        l2_lambda=0.05,
        learning_rate=0.05,
    )


    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, dataloader_train)
    w_model = np.append(
        model.output_layer.bias.detach().numpy()[0],
        model.output_layer.weight.detach().numpy(),
    )
    
    # trainer.test(model, dataloader_test)

    y_pred = model(X_test).view(-1)
    
    r2score = R2Score()
    score = r2score(y_pred, y_test).item()

    print(score)


    
    # %%
