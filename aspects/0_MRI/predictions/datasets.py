import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class MRIs(Dataset):
    """
    """

    def __init__(self, csv_path, variable, task, name):
        self._csv_path = csv_path
        self.variable = variable
        self.task = task
        self.name = name
        self.data = None
        self.labels = None
        self.data, self.labels, self.vital_status = MRIs.get_data_rna(
            self._csv_path, self.variable, self.task)

        self._get_feat_characteristics()

    def _get_feat_characteristics(self):
        self.num_classes = 1
        if self.task == "classification":
            self.num_classes = int(torch.max(self.labels).item()) + 1

        self.num_cases = self.data.size(0)
        self.num_features = self.data.size(1)

        feat_char = (self.num_features, self.num_classes)

        return feat_char

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]

        vital_status = self.vital_status[idx]

        return features, label, vital_status

    @staticmethod
    def get_data_rna(csv_path, variable, task):

        le = LabelEncoder()
        data = pd.read_csv(csv_path)
        features, labels = [], []
        vital_status = torch.tensor([0 for _ in range(data.shape[0])])

        features = data[[x for x in data.keys() if x.isdigit()]
                        ].values.astype(np.float32)
        features = torch.tensor(features)

        if task == "survival":
            labels = data["survival_months"]
            vital_status = torch.tensor(data["vital_status"]).float()
        else:
            labels = data[variable]
            if task == "classification":
                labels = le.fit_transform(labels)

        labels = torch.tensor(labels)

        return features, labels, vital_status


# MODIFIER POUR AVOIR UNE SEULE GROSSE BASE A SPLITTER EN 3
# PRENDRE EN INPUT 2 FICHIERS CORRESPONDANT :
#   - AUX DONNEES DE SURVIE
#   - AUX MR FEATURES
