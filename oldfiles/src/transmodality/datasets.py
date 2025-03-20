import os
from pathlib import Path
from os.path import join, isdir, basename
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torchio as tio
import glob


class CrossMod(Dataset):
    def __init__(self, data_path, modalities):

        self.data_path = [join(dp, "processed") for dp in data_path]
        self.cases = []
        for dp in self.data_path:
            self.cases.extend(
                [join(dp, c) for c in os.listdir(dp) if isdir(join(dp, c))])

        self.modalities = modalities
        self.n_mod = len(self.modalities)

        self.nifti_files = []
        self.corresp = {}
        for case in self.cases:
            mods = []
            for mod in self.modalities:
                mod += ".nii.gz"
                files = glob.glob(join(case, mod))
                self.nifti_files.extend(files)
                mods.extend(files)

            self.corresp[mods[0]] = mods[1]
            self.corresp[mods[1]] = mods[0]

        self.corresp = list(self.corresp.items())

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

        inp, out = self.corresp[idx]
        case_folder = str(Path(inp).parent)

        # case = inp.split("/")[-2]
        # mod = inp.split("/")[-1].rstrip(".nii.gz")
        inp = self.import_nifti(inp)
        out = self.import_nifti(out)

        return inp, out, case_folder



class HalfDataset(Dataset):
    def __init__(self, data_path, modalities):

        self.data_path = [join(dp, "processed") for dp in data_path]
        self.cases = []
        for dp in self.data_path:
            self.cases.extend(
                [join(dp, c) for c in os.listdir(dp) if isdir(join(dp, c))])

        self.modalities = modalities
        self.n_mod = len(self.modalities)

        self.nifti_files = []
        self.corresp = {}
        for case in self.cases:
            mods = []
            for mod in self.modalities:
                mod += ".nii.gz"
                files = glob.glob(join(case, mod))
                mods.extend(files)
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

        inp = self.nifti_files[idx]
        case_folder = str(Path(inp).parent)

        inp = self.import_nifti(inp)

        return inp, case_folder
