import nibabel as nib
from os.path import join, basename, isfile
from tqdm import tqdm
import multiprocessing
from glob import glob
import os

SLICE = 160
DATA = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR"
dataset = "UKBIOBANK"
data = join(DATA, dataset, "processed")



def crop_file(file):
    bname = basename(file)
    newname = bname.replace(".nii","_crop.nii")
    output_name = file.replace(bname, newname)
    print(file)

    if not isfile(output_name):
        try:
            img = nib.load(file)
            img.slicer[:,:,:SLICE].to_filename(output_name)
        except:
            print("removing :", file)
            os.remove(file)
    else:
        pass



if __name__=='__main__':

    files = [f for f in glob(data + "/*/*normT*.nii.gz") if not "_crop.nii" in f]

    with multiprocessing.Pool(10) as p:
        r = list(tqdm(p.imap(crop_file, files),
                        total = len(files), colour="yellow"))

