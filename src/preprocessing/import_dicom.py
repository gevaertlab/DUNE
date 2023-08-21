# %%
import os
from os.path import join, isdir, basename
import dicom2nifti
import multiprocessing
from tqdm import tqdm
from pathlib import Path


DATA_FOLDER = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/VESTSEG/original/Vestibular-Schwannoma-SEG"
OUTPUT_FOLDER = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/VESTSEG/images"


def ls_dir(path):
    dirs = [join(path, d) for d in os.listdir(path) if isdir(join(path, d))]
    return dirs





def process(case):

    T1 = Path(case).glob("**/*-t1*")
    T1 = [f for f in T1][0]
    T2 = Path(case).glob("**/*-t2*")
    T2 = [f for f in T2][0]

    caseid = basename(T1.parents[1])

    target = join(OUTPUT_FOLDER, caseid)
    os.makedirs(target, exist_ok=True)

    if T1:
        dicom2nifti.convert_directory(T1, target)
    if T2:
        dicom2nifti.convert_directory(T2, target)

    return


if __name__ == "__main__":
    cases = ls_dir(DATA_FOLDER)
    with multiprocessing.Pool(10) as p:
        r = list(tqdm(p.imap(process, cases), total=len(cases), colour='cyan'))



# %%
