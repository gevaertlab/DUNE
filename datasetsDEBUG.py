import os
from os.path import join, isdir
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torchio as tio


class BrainImages(Dataset):
    def __init__(self, dataset, data_path, modalities, mindims, whole_brain=True, transforms=None):

        self.dataset = dataset
        self.data_path = join(data_path, "processed")
        self.folders = [case for case in os.listdir(self.data_path) if isdir(join(self.data_path, case))]

        self.modalities = modalities
        self.transforms = transforms
        self.n_mod = len(self.modalities)
        self.depth, self.height, self.width = mindims
        self.whole_brain = whole_brain        

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        self.imgAddresses = []
        case = self.folders[idx]
        # try:
        

        for mod in self.modalities:
            mod = mod.lower() + ".nii"
            folderpath = join(self.data_path, self.folders[idx])

            filepath = [join(folderpath, f) for f in os.listdir(
                folderpath) if mod in f.lower()][0]
            self.imgAddresses.append(filepath)

        imgs = [nib.load(self.imgAddresses[i]) for i in range(self.n_mod)]
        rescale = tio.RescaleIntensity(out_min_max=(0,1.0))
        imgs = [rescale(i).get_fdata() for i in imgs]

        imgs = np.array(imgs, dtype=np.uint8).transpose((0, 3, 2, 1))
        imgs = torch.Tensor(imgs)


        # except:
        #     print("case is with problem : ", case)
        #     imgs = torch.Tensor(0)
        #     pass

        
        return imgs, case
