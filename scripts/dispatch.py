import os
import shutil
from os.path import join


INPUT = "../data/data_fusion/MR/from/"
OUTPUT = "../data/data_fusion/MR/dispatch2/"
modalities = ["t1", "t2", "t1Gd", "flair", "GlistrBoost"]



if __name__ == "__main__":
    case_dirs = [a for a in os.listdir(INPUT) if os.path.isdir(INPUT + a)]
    dic_mod = {mod: [] for mod in modalities}
    for _, case in enumerate(case_dirs):
        for mod in modalities:
            dic_mod[mod].extend([join(INPUT, case, f)
                                 for f in os.listdir(INPUT + case) if f"{mod}.nii" in f])


    print(dic_mod["t1"])