import os
import nibabel as nib
import numpy as np
from torch import tensor as t
from os.path import join
import shutil
from alive_progress import alive_bar

ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/"
bugs = []

with alive_bar(len(os.listdir(ROOT))) as bar:
    for case in os.listdir(ROOT):
        casedir = join(ROOT, case)
        try:
            t1 = [join(casedir, f)
                  for f in os.listdir(casedir) if "T1" in f][0]
            flair = [join(casedir, f)
                     for f in os.listdir(casedir) if "FLAIR" in f][0]
            t1 = np.array(nib.load(t1).get_fdata().transpose((2, 1, 0)))
            t2f = np.array(nib.load(flair).get_fdata().transpose((2, 1, 0)))
        except IndexError:
            print("appending : ", case)
            bugs.append(case)
        try:
            stack = np.array([t1, t2f])
            fin = t(stack)
        except TypeError as e:
            print("appending : ", case)
            bugs.append(case)
            pass

        bar()

print(bugs)
for case in bugs:
    print("removing ", case)
    shutil.rmtree(join(ROOT, case))
