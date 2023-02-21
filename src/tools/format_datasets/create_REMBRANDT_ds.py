# %%
import os
from os import path
import shutil



ROOT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
INPUT = path.join(ROOT_DIR, "data/MR/REMBRANDT/REMBRANDTVerified") 
OUTPUT = path.join(ROOT_DIR, "data/MR/REMBRANDT/images") 


if __name__=='__main__':
    os.chdir(ROOT_DIR)

    cases = [f for f in os.listdir(INPUT) if path.isdir(path.join(INPUT, f))]


    for _, case in enumerate(cases):
        case_dir = path.join(INPUT, case)

        target_dir = path.join(OUTPUT, case[:-1])

        sequences = ["t1-post_pre_bcorr_brain.nii","flair_t1_bcorr_brain.nii", "tumor.nii.gz"]
        newnames = [f"{case[:-1]}_t1Gd.nii", f"{case[:-1]}_flair.nii", f"{case[:-1]}_segm.nii.gz"]

        for file, newfile in zip(sequences, newnames):

            img = path.join(INPUT, case, file)
            if path.isfile(img):
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy(path.join(INPUT, case, file), path.join(target_dir, newfile))
            else:
                print("skipping ", img)

        

# %%

