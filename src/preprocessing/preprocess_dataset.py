import os
from os.path import join, basename
from glob import glob
import shutil
import multiprocessing
from tqdm import tqdm

START, STOP = 0, 20000

DATA_ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR"
DATASET = "VESTSEG"
MODS = {"T1": "_t1_", "T2": "_t2_"}

standardize = "src/preprocessing/brain_standardization.sh"
correct = "src/preprocessing/bias_correction.sh"
extract = "src/preprocessing/brain_extract.sh"


brainExtraction = True
biasCorrection = True
brainStandardization = True


def ls_dir(path):
    dirs = [join(path, d) for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))]
    return dirs


def copy_files(case, files):
    t1 = files["T1"]
    t2 = files["T2"]

    output_dir = join(OUTPUT_DIR, basename(case))
    os.makedirs(output_dir, exist_ok=True)

    new_t1 = t1.replace(INPUT_DIR, OUTPUT_DIR).replace(
        basename(t1), "_init_"+MODS["T1"]+".nii.gz")
    new_t2 = t2.replace(INPUT_DIR, OUTPUT_DIR).replace(
        basename(t2), "_init_"+MODS["T2"]+".nii.gz")

    shutil.copy(t1, new_t1)
    shutil.copy(t2, new_t2)

    return output_dir, new_t1, new_t2


def brain_extraction(inp, mod):

    output = inp.replace(basename(inp), mod)

    os.system(f'bash {extract} -i {inp} -o {output}')

    try:
        os.remove(f'{output}_BrainExtractionPrior0GenericAffine.mat')
        os.remove(f'{output}_BrainExtractionMask.nii.gz')
        os.remove(inp)
    except:
        pass
    output = f'{output}_BrainExtractionBrain.nii.gz'

    return output


def bias_correction(inp, mod):
    print("Correction")
    output = inp.replace(basename(inp), mod+".nii.gz")

    os.system(f'bash {correct} -i {inp} -o {output}')
    os.remove(inp)

    return output


def brain_standardization(t1, t2, output_dir):
    print("Normalization")

    cmd = f'bash {standardize} -t {t1} -u {t2} -o {output_dir}'
    os.system(cmd)

    return


def preprocess(case):
    print(f"Processing {basename(case)}...")

    files = {k: glob(case + f"/*{v}*nii.gz")[0] for k, v in MODS.items()}

    output_dir, new_t1, new_t2 = copy_files(case, files)
    print(files)

    if brainExtraction:
        new_t1 = brain_extraction(new_t1, mod="T1")
        new_t2 = brain_extraction(new_t2, mod="T2")

    if biasCorrection:
        new_t1 = bias_correction(new_t1, mod="T1")
        new_t2 = bias_correction(new_t2, mod="T2")

    if brainStandardization:
        brain_standardization(new_t1, new_t2, output_dir)

    shutil.rmtree(join(output_dir, "warp"))

    return


def is_complete(case):

    try:
        files = os.listdir(case.replace("images", "processed"))
        return "normT1.nii.gz" in files and "normT2.nii.gz" in files
    except FileNotFoundError:
        return False


if __name__ == "__main__":

    INPUT_DIR = join(DATA_ROOT, DATASET, "images")
    OUTPUT_DIR = join(DATA_ROOT, DATASET, "processed")
    cases = ls_dir(INPUT_DIR)

    cases = cases[START: STOP]
    cases = [f for f in cases if not is_complete(f)]

    with multiprocessing.Pool(10) as p:
        r = list(tqdm(p.imap(preprocess, cases),
                      total=len(cases), colour="yellow"))
