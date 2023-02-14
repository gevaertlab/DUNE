# %%
import os
from os.path import join, isdir, exists
import glob
import itertools
import shutil
from dicom2nifti import convert_directory

ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/"
SOURCE = "/labs/gevaertlab/data/radiology/StanfordGBM/2018-03-05/GBM"
DESTINATION = join(ROOT, "data/MR/STANFORD")


def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs


if __name__ == "__main__":
    patients = ls_dir(SOURCE)

    patients_dict = {}

    for p in patients[0:1]:
        t1_patterns = [
            "*AX*T1*POST*", "*AX*T1*GAD*", "*AX*T1*C*", "*AX*FSPGR*C*",
            "*Ax*T1*C*", "*AX*T1*post*", "*AX*T1*gad*", "*GAD*T1*AX*", "*AX*T1*Post*", "*T1*AX*POST*", "AX POST - 901", "*C AX FSPGR BRAVO*", "*C*AX*T1*", "*Ax*GSP*C*", "*Ax*BRAVO*C*", "*POST*AX*T1*", "*POST*3DFSP*", "*C*Ax*T1*"]
        flair_patterns = ["*AX*FLAIR*", "*FLAIR*AX*", "*T2*FLAIR*","*AX*T2*FLAIR*", "*Ax*FLAIR*", "*FLAIR_long*", "*flair_ax*"]

        t1 = [glob.glob(join(SOURCE, p, "*/",t)) for t in t1_patterns]
        t1 = list(itertools.chain.from_iterable(list(t1)))
        t1 = [f for f in t1 if "FLAIR" not in f]


        flair = [glob.glob(join(SOURCE, p, "*/",t)) for t in flair_patterns]
        flair = list(itertools.chain.from_iterable(flair))
        flair = [f for f in flair if "T1" not in f]
        flair = [f for f in flair if "COR" not in f]
        
        if flair and t1:
            flair = flair[0]
            t1 = t1[0]
            files = [t1, flair]

            patients_dict[p] = files

    for p, f in patients_dict.items():
        root_dest_folder = join (DESTINATION, p.split(" ")[-1])
        os.makedirs(root_dest_folder, exist_ok=True)
        print("\n", p)
        for seq in f:
            sequence = "FLAIR" if "FLAIR" in seq  else "T1Gd"
            dest_folder = join(root_dest_folder, sequence)
            # shutil.copytree(seq,  dest_folder)

            # NIFTI CONVERSTION
            convert_directory(dest_folder, dest_folder)
            filename = [f for f in os.listdir(dest_folder) if "nii" in f][0]
            newfilename = join(root_dest_folder, sequence) + ".nii.gz"
            os.rename(join(dest_folder, filename), newfilename)
            shutil.rmtree(dest_folder)


# %%