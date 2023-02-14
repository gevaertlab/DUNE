# %%
import os
from os.path import join, isfile
import nibabel as nib
from brainextractor import BrainExtractor
import torchio as tio
from tqdm import tqdm

ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/"
DATA = join(ROOT, "data/MR/STANFORD")



def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs





if __name__ == "__main__":
        
    patients= ls_dir(DATA)

    for patient in tqdm(patients[0:1]):

        imgs= [f for f in os.listdir(join(DATA, patient)) if f.endswith("nii.gz") if not "mask" in f]

        for seq in imgs:
            print(seq)
            img_path = join(DATA, patient, seq)
            img = nib.load(img_path)
            resample = tio.Resample(1, image_interpolation='bspline')

            img = resample(img)
           
            # Brain extraction
            mask_path = join(DATA, patient, f"{seq[:-7]}-mask.nii.gz")
            try:
                bet = BrainExtractor(img = img)
                bet.run(iterations=1000)
                bet.save_mask(mask_path)
            except:
                pass

            # Masking and exporting
            img_data = img.get_fdata()

            mask = nib.load(mask_path).get_fdata()
            masked_data = img_data * mask

            output = nib.Nifti1Image(masked_data, header=img.header, affine=img.affine)

            output_path = join(DATA, patient, f"0-masked-{seq}")
            nib.save(output, output_path)


# %%
