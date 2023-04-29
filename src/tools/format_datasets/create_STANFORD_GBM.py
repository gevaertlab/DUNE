# %%
import os
from os.path import join, isdir, exists
import glob
import itertools
import shutil
from dicom2nifti import convert_directory
import pandas as pd
from tqdm import tqdm

ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/"
SOURCE = "/labs/gevaertlab/data/radiology/StanfordGBM/2018-03-05/GBM"
DESTINATION = join(ROOT, "data/MR/STANFORD/images")
FOLDERNAMES = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/STANFORD/StanfordGBM.xlsx"

def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs


if __name__ == "__main__":
    patients = ls_dir(SOURCE)

    patients_dict = {}

    patients_dict = pd.read_excel(FOLDERNAMES, index_col=0)
    patients_dict = {col:list(patients_dict[col]) for col in patients_dict.columns if "EXCLUDE" not in list(patients_dict[col])}

    # for p in patients[0:1]:


    #     t1 = [glob.glob(join(SOURCE, p, "*/",t)) for t in t1_patterns]
    #     t1 = list(itertools.chain.from_iterable(list(t1)))
    #     t1 = [f for f in t1 if "FLAIR" not in f]


    #     flair = [glob.glob(join(SOURCE, p, "*/",t)) for t in flair_patterns]
    #     flair = list(itertools.chain.from_iterable(flair))
    #     flair = [f for f in flair if "T1" not in f]
    #     flair = [f for f in flair if "COR" not in f]
        
    #     if flair and t1:
    #         flair = flair[0]
    #         t1 = t1[0]
    #         files = [t1, flair]

    #         patients_dict[p] = files

    for p, f in tqdm(patients_dict.items()):
        root_dest_folder = join (DESTINATION, p.split(" ")[-1])
        os.makedirs(root_dest_folder, exist_ok=True)

        T1Gd, flair = f

        T1_folder = glob.glob(join(SOURCE, p, "*/",T1Gd))[0]

        flair_folder = glob.glob(join(SOURCE, p, "*/",flair))[0]

        sequences = {"T1Gd":T1_folder, "FLAIR":flair_folder}
        # NIFTI CONVERSTION
        for seq, path in sequences.items():
            dest_folder = join(root_dest_folder, seq)
            shutil.copytree(path, dest_folder)
            convert_directory(dest_folder, dest_folder)
            filename = [f for f in os.listdir(dest_folder) if "nii" in f][0]
            newfilename = join(root_dest_folder, seq) + ".nii.gz"
            os.rename(join(dest_folder, filename), newfilename)
            shutil.rmtree(join(dest_folder))



# %%