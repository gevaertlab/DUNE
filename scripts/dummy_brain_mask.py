import os
from os.path import isdir, exists, join
import nibabel as nib
from tqdm import tqdm



DATA_DIR = "./data/MR/UKBIOBANK"
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

pool = Pool(20)
pool.map(create_mask,tqdm(cases))