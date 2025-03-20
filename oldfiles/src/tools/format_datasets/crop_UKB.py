from random import choices
import nibabel as nib
import numpy as np
import os
from os.path import join
from tqdm import tqdm
from multiprocessing import Pool

np.random.seed(123)
# os.chdir("/home/tbarba/projects/MultiModalBrainSurvival/")
DATA =  "data/MR"
DATASET = "UKBIOBANK"
OUTPUT_DIR = join(DATA, "UKB_crop/images")
BBOX_DIM = [68, 94, 74]
NUM_CENTERS = 3

def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs



def import_modality(path, pattern):
    files = os.listdir(path)
    mod = [f for f in files if f"{pattern.lower()}.nii" in f.lower()][0]
    mod = join(path, mod)
    mod = nib.load(mod)

    return mod


def crop_nifti(nifti, center, bbox_dim):

    width, height, depth = bbox_dim

    maxX, maxY, maxZ = nifti.shape
    centerX, centerY, centerZ = center
    
    halfw = width //2
    halfh = height //2
    halfd = depth //2

    lowX = max(centerX - halfw, 0)
    highX = width if lowX == 0 else min(centerX + halfw, maxX)
    # highX = min(centerX + halfw, maxX)
    # lowX = maxX-width if highX == maxX else lowX
    
    lowY = max(centerY - halfh, 0)
    highY = height if lowY == 0 else min(centerY + halfh, maxY)
    # highY = min(centerY + halfh, maxY)
    # lowY = maxX-height if highY == maxY else lowY

    lowZ = max(centerZ - halfd, 0)
    highZ = depth if lowZ == 0 else min(centerZ + halfd, maxZ)
    # highZ = min(centerZ + halfd, maxZ)
    # lowZ = maxZ-depth if highZ == maxZ else lowZ


    # Slicer
    cropped_nifti = nifti.slicer[lowX:highX, lowY:highY, lowZ:highZ]

    return cropped_nifti


def process(eid):

    eid_path = join(images_path, eid)
    t1 = import_modality(eid_path, modalities[0])
    flair = import_modality(eid_path, modalities[1])
    mask = import_modality(eid_path, modalities[2])

    # coordonnées des points positifs du mask
    coords = tuple(zip(*np.where(mask.get_fdata() ==1)))

    random_centers = choices(coords, k=NUM_CENTERS)

    for center in random_centers:

        x,y,z = center
        output_eid = join(OUTPUT_DIR, f"{eid}_c{x}{y}{z}")
        os.makedirs(output_eid, exist_ok=True)

        t1_cropped = crop_nifti(t1, center, BBOX_DIM)
        flair_cropped = crop_nifti(flair, center, BBOX_DIM)
        nib.save(t1_cropped, join(output_eid, 'T1c_crop.nii.gz'))
        nib.save(flair_cropped, join(output_eid, 'FLAIR_crop.nii.gz'))




if __name__ == "__main__":
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_path = join(DATA, DATASET, "images")
    modalities=["t1","FLAIR", "mask"]
    eids = ls_dir(images_path)

    with Pool(6) as p:
        list(tqdm(p.imap(process, eids), total=len(eids), colour="blue"))
