# %%
import os
from os.path import isdir, exists, join
import nibabel as nib
from tqdm import tqdm
from nilearn.masking import compute_brain_mask
import multiprocessing
import sys

# %%



DATASET = sys.argv[1]
DATA_DIR = f"/home/tbarba/projects/MultiModalBrainSurvival/data/MR/{DATASET}/processed"
cases = [f for f in os.listdir(DATA_DIR) if isdir(join(DATA_DIR, f))]



def create_mask(case):
    if not exists(join(DATA_DIR, case, "brain_mask.nii.gz")):
        t1 = join(DATA_DIR, case,"normT1_crop.nii.gz")
        mask = compute_brain_mask(t1, threshold=0)
        nib.save(mask, join(DATA_DIR, case, "brain_mask.nii.gz"))

    else:
        print(f"\n{case} Already masked.")


def main():
    with multiprocessing.Pool(10) as p:
        r = list(tqdm(p.imap(create_mask, cases), total = len(cases), colour='cyan'))


if __name__ == "__main__":
    main()