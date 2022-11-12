import os
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torch

from torchvision import transforms
import pandas as pd

class MRDataset(Dataset):
    """
    """

    def __init__(self, csv_path):


        self._csv_path = csv_path
        self.data = None
        self._preprocess()

    def _preprocess(self):
        self.data = MRDataset.get_data_rna(self._csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item['idx']=idx #test
        return item

    @staticmethod
    def get_data_rna(csv_path):
        
        dataset = []

        data = pd.read_csv(csv_path)
        print("imported !")
        for _, row in data.iterrows():

            mr_data = row[[x for x in row.keys() if x not in ["case","sex", "age","survival_months","vital_status"]]].values.astype(np.float32)
            mr_data = torch.tensor(mr_data, dtype=torch.float32)

            row = row[[x for x in row.keys() if x in ["case","sex", "age","survival_months","vital_status"]]].to_dict()
            row['mr_data'] = mr_data
            try:
                row['vital_status'] = np.float32(row['vital_status'])
                row['survival_months'] = np.float32(row['survival_months'])
            except KeyError:
                pass

            try:
                row['sex'] = np.float32(row['sex'])
            except KeyError:
                pass
            
            item = row.copy()
            dataset.append(item)

        return dataset



# MODIFIER POUR AVOIR UNE SEULE GROSSE BASE A SPLITTER EN 3
# PRENDRE EN INPUT 2 FICHIERS CORRESPONDANT : 
#   - AUX DONNEES DE SURVIE
#   - AUX MR FEATURES