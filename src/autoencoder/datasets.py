import os
from os.path import join, isdir, basename
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torchio as tio

class BrainImages(Dataset):
    def __init__(self, data_path, modalities):
        
        self.data_path = [join(dp, "processed") for dp in data_path]
        self.cases = []
        for dp in self.data_path:
            self.cases.extend([join(dp, c) for c in os.listdir(dp) if isdir(join(dp, c))])

        self.modalities = modalities
        self.n_mod = len(self.modalities)
        # self.depth, self.height, self.width = mindims

        print(len(self.cases))


    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        self.NIFTIs = []
        case = self.cases[idx]
        
        for mod in self.modalities:
            mod = mod.lower() + ".nii"
            nifti_files = [join(case, f) for f in os.listdir(case) if mod in f.lower()][0]
            self.NIFTIs.append(nifti_files)
        
        rescale = tio.RescaleIntensity(out_min_max=(0,1))

        imgs = [nib.load(self.NIFTIs[i]) for i in range(self.n_mod)]
        imgs = [rescale(i).get_fdata() for i in imgs]
        imgs = np.array(imgs, dtype=np.float16).transpose((0, 3, 2, 1))
        imgs = torch.Tensor(imgs)

        
        return imgs, basename(case)
