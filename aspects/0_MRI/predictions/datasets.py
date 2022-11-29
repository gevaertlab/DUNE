import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class MRDataset(Dataset):
    """
    """

    def __init__(self, csv_path, variable, task):
        self._csv_path = csv_path
        self.variable = variable
        self.task = task
        self.data = None
        self.labels = None
        self.num_classes = 1
        self._preprocess()

    def _preprocess(self):
        self.data, self.labels = MRDataset.get_data_rna(self._csv_path, self.variable, self.task)
        if self.task == "classification":
            self.num_classes = int(torch.max(self.labels).item()) +1

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        return features, label

    @staticmethod
    def get_data_rna(csv_path, variable, task):

        le = LabelEncoder()
        data = pd.read_csv(csv_path)            
        features, labels = [], []

        features = data[[x for x in data.keys() if x.isdigit()]].values.astype(np.float32)
        features = torch.tensor(features)
        labels = data[variable]

        if task == "classification":
            labels = le.fit_transform(labels)

        labels = torch.tensor(labels)

        return features, labels



# MODIFIER POUR AVOIR UNE SEULE GROSSE BASE A SPLITTER EN 3
# PRENDRE EN INPUT 2 FICHIERS CORRESPONDANT : 
#   - AUX DONNEES DE SURVIE
#   - AUX MR FEATURES