# %%
import os
from os.path import isdir, exists, join
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
DATA_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/images"
cases = [f for f in os.listdir(DATA_DIR) if isdir(join(DATA_DIR, f))]

def create_mask(case):
    if not exists(join(DATA_DIR, case, "brain_mask.nii.gz")):
        print(f"\nProcessing {case}")
        t1_path = [join(DATA_DIR, case,f) for f in os.listdir(join(DATA_DIR, case)) if "t1" in f.lower()][0]
        img = nib.load(t1_path)
        brain_mask = (img.get_fdata() > 1)
        brain_mask = brain_mask.astype(float)
        final_brain_mask = nib.Nifti1Image(brain_mask, img.affine)  
        nib.save(final_brain_mask, join(DATA_DIR, case, "brain_mask.nii.gz"))
    else:
        print(f"\n{case} Already masked.")

from multiprocessing import Pool

# pool = Pool(5)
# pool.map(create_mask,tqdm(cases))



# %%

DATA_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/REMBRANDT/images"
cases = [f for f in os.listdir(DATA_DIR) if isdir(join(DATA_DIR, f))]
mask_pattern = "segm.nii"


# Create binary brain tumor mask
for case in cases:

    mask = [f for f in os.listdir(join(DATA_DIR, case)) if  mask_pattern in f][0]
    mask = join(DATA_DIR, case, mask)    
    mask = nib.load(mask)
    new_mask = mask.get_fdata() > 0

    final_brain_mask = nib.Nifti1Image(new_mask.astype(float), mask.affine)  
    nib.save(final_brain_mask, join(DATA_DIR, case, "SegmBinary.nii.gz"))

# %%
