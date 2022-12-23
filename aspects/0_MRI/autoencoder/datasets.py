import os
import torch
import nibabel as nib
from PIL import Image
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

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        self.imgAddresses = []
        for mod in self.modalities:
            mod = mod.lower() + ".nii"
            folderpath = os.path.join(self.data_path, self.folders[idx])
            filepath = [os.path.join(folderpath, f) for f in os.listdir(
                folderpath) if mod in f.lower()][0]
            self.imgAddresses.append(filepath)
        case = self.folders[idx]
        imgsPre = []
        for i in range(0, self.n_mod):
            nifti = nib.load(self.imgAddresses[i])
            # self.voxel_size = nifti.header.get_zooms()
            resample = tio.Resample(1, image_interpolation='bspline')
            rescale = tio.RescaleIntensity(out_min_max=(0,255.0))

            nifti = resample(nifti)
            nifti = rescale(nifti)
            temp = nifti.get_fdata()
            temp = temp.transpose((2, 1, 0))
            temp = np.array(temp)
            temp = temp.astype(np.uint8)

            imgsPre.append(temp.copy())

        imgsPre = np.array(imgsPre, dtype=np.uint8)
        imgsPre = torch.tensor(imgsPre)

        centerZ, centerY, centerX = BrainImages.find_brain_center(imgsPre)

        startSliceZ = int(
            centerZ - self.depth/2) if centerZ > self.depth/2 else 0
        endSliceZ = int(startSliceZ) + self.depth

        startSliceY = int(
            centerY - self.height/2) if centerY > self.height/2 else 0
        endSliceY = int(startSliceY) + self.height

        startSliceX = int(
            centerX - self.width/2)if centerX > self.width/2 else 0
        endSliceX = int(startSliceX) + self.width

        imgsPil = torch.zeros(
            [self.n_mod, self.depth, self.height, self.width])

        for i in range(0, self.n_mod):
            for z in range(startSliceZ, endSliceZ):
                t = imgsPre[i, z, startSliceY:endSliceY,
                            startSliceX:endSliceX].numpy()
                t = Image.fromarray(t)

                if self.dataset in ("UKBIOBANK", "REMBRANDT"):
                    t = t.rotate(angle=180)

                imgsPil[i, z-startSliceZ, :,
                        :] = transforms.ToTensor()(t)

        imgs = imgsPil[:, :, :, :]
        
        return imgs, case
