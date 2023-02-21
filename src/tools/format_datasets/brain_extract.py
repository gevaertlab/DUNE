# %%
import os
from os.path import join, isfile
import nibabel as nib
from brainextractor import BrainExtractor
import torchio as tio
from tqdm import tqdm
from nilearn.image import resample_img, crop_img

ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/"
DATA = join("data/MR/SCHIZO/images/selected")
UKB_AFFINE = nib.load(join(ROOT, "data/MR/UKBIOBANK/images/1000739/1000739_T1.nii.gz")).affine


def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs


# img_path = join(DATA, "T1MCICS.nii.gz")
# img = nib.load(img_path)
# bet = BrainExtractor(img = img)
# bet.run(iterations=1000)

# mask_path = join(ROOT, "mask.nii.gz")

# bet.save_mask(mask_path)

# img_data = img.get_fdata()
# mask = nib.load(mask_path).get_fdata()
# masked_data = img_data * mask
# output = nib.Nifti1Image(masked_data, header=img.header, affine=img.affine)

# output_path = join(DATA, "extracted.nii.gz")
# nib.save(output, output_path)


if __name__ == "__main__":
        
    patients= ls_dir(DATA)

    for patient in tqdm(patients):

        imgs= [f for f in os.listdir(join(DATA, patient)) if f.endswith("nii.gz") if not "mask" in f]

        for seq in imgs:            
            seq_name = "T1w" if "T1w" in seq else "T2w"
 
            img_path = join(DATA, patient, seq)
            img = nib.load(img_path)

            # Brain extraction
            mask_path = join(DATA, patient, f"{seq[:-7]}-mask.nii.gz")
            if not isfile(mask_path):
                try:
                    bet = BrainExtractor(img = img)
                    bet.run(iterations=1000)
                    bet.save_mask(mask_path)
                except:
                    pass             

            # Masking and exporting
            mask = nib.load(mask_path).get_fdata()
            img_data = img.get_fdata()
            masked_data = img_data * mask

            # Resampling and cropping
            print("Resampling and cropping")
            masked_nifti = nib.Nifti1Image(masked_data, header=img.header, affine=img.affine)
            resampled_nifti = resample_img(masked_nifti, target_affine=UKB_AFFINE)
            resampled_nifti = crop_img(resampled_nifti)

            output_path = join(DATA, patient, f"{patient}-masked-{seq_name}.nii.gz")
            nib.save(resampled_nifti, output_path)


# # %%
