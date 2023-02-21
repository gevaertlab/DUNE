# %%
import os
from os.path import join
import nibabel as nib
import torchio as tio
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import shuffle

def ls_dir(path):
    dirs = (d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    return dirs

def calc_brain_lim(nifti_array):
    """
    Finds dims of brain box
    """
    D, H, W = nifti_array.shape

    # calc brain limits
    z_limits = [z for z in range(D) if nifti_array[z, :, :].sum() > 10]
    depth = z_limits[-1] - z_limits[0]
    
    y_limits = [y for y in range(H) if nifti_array[:, y, :].sum() > 10]
    height = y_limits[-1] - y_limits[0]

    
    x_limits = [x for x in range(W) if nifti_array[:, :, x].sum() > 10]
    width = x_limits[-1] - x_limits[0]

    return depth, height, width


ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR"
DATASETS = ["UKBIOBANK", "TCGA"]


if __name__ == "__main__":
    
    for dataset in DATASETS:

        print(dataset)
        data_path = join(ROOT, dataset, "images")
        full_depths, full_heights, full_widths = [], [], []
        brain_depths, brain_heights, brain_widths = [], [], []
        resample = tio.Resample(1, image_interpolation='bspline')
        cases = ls_dir(data_path)

        for idx, case in tqdm(enumerate(cases)):
            img = [i for i in os.listdir(join(data_path, case)) if "nii" in i][0]
            img = nib.load(join(data_path, case, img))
            img = resample(img)
            img = img.get_fdata()
            img = img.transpose((2, 1, 0))

            full_depth, full_height, full_width = img.shape
            brain_depth, brain_height, brain_width = calc_brain_lim(img)

            full_depths.append(full_depth)
            full_heights.append(full_height)
            full_widths.append(full_width)
            
            brain_depths.append(brain_depth)
            brain_heights.append(brain_height)
            brain_widths.append(brain_width)

            
            # Computing limits

            depth_limit = np.min(full_depths)
            height_limit = np.min(full_heights)
            width_limit = np.min(full_widths)
            D_mean, D_std = np.mean(brain_depths), np.std(brain_depths)
            H_mean, H_std = np.mean(brain_heights), np.std(brain_heights)
            W_mean, W_std = np.mean(brain_widths), np.std(brain_widths)

            
            proposed_D_mindim = int(D_mean + 3*D_std)
            retained_D_mindim = min(depth_limit, proposed_D_mindim )
            proposed_H_mindim = int(H_mean + 3*H_std)
            retained_H_mindim = min(height_limit, proposed_H_mindim )
            proposed_W_mindim = int(W_mean + 3*W_std)
            retained_W_mindim = min(width_limit, proposed_W_mindim )


            results = {
                "brain_depths":np.rint([D_mean, D_std, proposed_D_mindim, depth_limit, retained_D_mindim]).astype(int),
                "brain_heights":np.rint([H_mean, H_std, proposed_H_mindim, height_limit, retained_H_mindim]).astype(int),
                "brain_widths":np.rint([W_mean, W_std, proposed_W_mindim, width_limit, retained_W_mindim]).astype(int),
                }
            
            results = pd.DataFrame.from_dict(results, orient="index", columns=["mean","std","mean+3std", "min_limit", "retained_mindims"])
            results["sample"] = idx+1
            results.index.name = "dimension"
            results.to_csv(join(ROOT, dataset, "min_dims.csv"), index=True)



# %%
