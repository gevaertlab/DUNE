import os
import torch
import nibabel as nib
from nilearn.image import resample_img, crop_img, concat_imgs, get_data
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage
import torchio as tio

class BrainImages(Dataset):
    def __init__(self, dataset, data_path, modalities, mindims, transforms=None):

        self.dataset = dataset
        self.data_path = data_path
        self.folders = [case for case in os.listdir(
            data_path) if os.path.isdir(os.path.join(data_path, case))]

        self.modalities = modalities
        self.transforms = transforms
        self.n_mod = len(self.modalities)
        self.depth, self.height, self.width = mindims
        
    @staticmethod
    def find_brain_center(stack):
        """
        Find coordinates of the brain center
        """
        sequence = stack[0]
        D, H, W = sequence.shape
        z_limits = [z for z in range(D) if sequence[z, :, :].sum().item() > 0]
        y_limits = [y for y in range(H) if sequence[:, y, :].sum().item() > 0]
        x_limits = [x for x in range(W) if sequence[:, :, x].sum().item() > 0]

        center = (
            int((z_limits[-1] + z_limits[0])/2),
            int((y_limits[-1] + y_limits[0])/2),
            int((x_limits[-1] + x_limits[0])/2))

        return center
    
    def func(self, nifti_path):
        img = nib.load(nifti_path)
        resample = tio.Resample(1, image_interpolation="bspline")
        img = resample(img)
        init = img.affine
        img = crop_img(img)
        img = resample_img(img, target_shape=[250,250,250], target_affine=init)
        rescale = tio.RescaleIntensity(out_min_max=(0,255.0))
        brain_center = BrainImages.find_brain_center(stacked_mod)
        img = rescale(img)
        return img, brain_center

    def __len__(self):
        return len(self.folders)


    def __getitem__(self, idx):

        self.imgAddresses = []
        for mod in self.modalities:
            mod = mod.lower()
            folderpath = os.path.join(self.data_path, self.folders[idx])

            filepath = [os.path.join(folderpath, f) for f in os.listdir(
                folderpath) if mod in f.lower()][0]
            self.imgAddresses.append(filepath)

        case = self.folders[idx]
        modalities = []
        for i in range(0, self.n_mod):
            nifti = self.imgAddresses[i]
            img, center = self.func(nifti)
            modalities.append(img)
            if i==0:
                centerZ, centerY, centerX = center

        stacked_mod = concat_imgs(modalities)
        stacked_mod = get_data(stacked_mod)
        stacked_mod = stacked_mod.transpose((3,2,1,0))


        stacked_mod = stacked_mod[:, 
                                  centerZ-self.depth//2:centerZ+self.depth//2,
                                  centerY-self.height//2:centerY+self.height//2,
                                  centerX-self.width//2:centerX+self.width//2]
        print(case, stacked_mod.shape)
        imgs = torch.tensor(stacked_mod)


        return imgs, case
