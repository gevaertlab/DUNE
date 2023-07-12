import os
from os.path import join, isdir, basename
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torchio as tio
import glob


class BrainImages(Dataset):
    def __init__(self, data_path, modalities):

        self.data_path = [join(dp, "processed") for dp in data_path]
        self.cases = []
        for dp in self.data_path:
            self.cases.extend(
                [join(dp, c) for c in os.listdir(dp) if isdir(join(dp, c))])

        self.modalities = modalities
        self.n_mod = len(self.modalities)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        self.NIFTIs = []
        case = self.cases[idx]

        for mod in self.modalities:
            nifti_files = glob.glob(join(case, f"{mod}.nii.gz"))
            self.NIFTIs.extend(nifti_files)
            # mod = mod.lower() + ".nii"
            # nifti_files = [join(case, f)for f in os.listdir(case) if mod in f.lower()][0]

        rescale = tio.RescaleIntensity(out_min_max=(0, 1))

        imgs = [nib.load(self.NIFTIs[i]) for i in range(self.n_mod)]
        imgs = [rescale(i).get_fdata() for i in imgs]
        imgs = np.array(imgs, dtype=np.float16).transpose((0, 3, 2, 1))
        imgs = torch.Tensor(imgs)

        return imgs, basename(case)


class SingleMod(Dataset):
    def __init__(self, data_path, modalities):

        self.data_path = [join(dp, "processed") for dp in data_path]
        self.cases = []
        for dp in self.data_path:
            self.cases.extend(
                [join(dp, c) for c in os.listdir(dp) if isdir(join(dp, c))])

        self.modalities = modalities
        self.n_mod = len(self.modalities)

        self.nifti_files = []
        for case in self.cases:
            for mod in self.modalities:
                mod += ".nii.gz"
                files = glob.glob(join(case, mod))
                self.nifti_files.extend(files)

    def __len__(self):

        return len(self.nifti_files)

    def import_nifti(self, nifti):

        imgs = nib.load(nifti)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        imgs = rescale(imgs).get_fdata()
        imgs = np.array(imgs, dtype=np.float16)
        imgs = np.expand_dims(imgs, axis=0)  # adding channel dim
        imgs = imgs.transpose((0, 3, 2, 1))  # reorder channels
        imgs = torch.Tensor(imgs)

        return imgs

    def __getitem__(self, idx):

        nifti = self.nifti_files[idx]

        case = nifti.split("/")[-2]
        mod = nifti.split("/")[-1].rstrip(".nii.gz")
        imgs = self.import_nifti(nifti)

        return imgs, f"{case}__{mod}"