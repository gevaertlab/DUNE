# %%
import os
from os.path import join, isfile
import nibabel as nib
from brainextractor import BrainExtractor
import torchio as tio
from tqdm import tqdm
from nilearn.image import resample_img, crop_img

ROOT = "/home/tbarba/projects/MultiModalBrainSurvival/"
DATA = join("data/MR/SCHIZO/images")
UKB_AFFINE = nib.load(join(ROOT, "data/MR/UKBIOBANK/images/1000739/1000739_T1.nii.gz")).affine


def ls_dir(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs


# img_path = join(DATA, "T1MCICS.nii.gz")
# img = nib.load`(img_path)
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

    for patient in tqdm(patients, colour="blue"):

        print("\n", patient)
        T1= [f for f in os.listdir(join(DATA, patient)) if f.endswith("nii.gz") if not "mask" in f if "T1w"in f][0]
        T2= [f for f in os.listdir(join(DATA, patient)) if f.endswith("nii.gz") if not "mask" in f if "T2w"in f][0]

        imgs = [T1, T2]

        for seq in imgs:            
            seq_name = "T1w" if "T1w" in seq else "T2w"
 
            img_path = join(DATA, patient, seq)
            img = nib.load(img_path)
            img = resample_img(img, target_affine=UKB_AFFINE)


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
            output_path = join(DATA, patient, f"{patient}-masked-{seq_name}.nii.gz")

            if not isfile(output_path):
                mask = nib.load(mask_path).get_fdata()
                img_data = img.get_fdata()
                masked_data = img_data * mask

                # Resampling and cropping
                print("Resampling and cropping")
                masked_nifti = nib.Nifti1Image(masked_data, header=img.header, affine=img.affine)
                resampled_nifti = resample_img(masked_nifti, target_affine=UKB_AFFINE)
                # resampled_nifti = crop_img(res`mpled_nifti)

                if seq_name =="T2w":
                    T1w = nib.load(join(DATA, patient, f"{patient}-masked-T1w.nii.gz"))
                    resampled_nifti = resample_img(resampled_nifti, target_shape= T1w.shape, target_affine=T1w.affine)

            nib.save(resampled_nifti, output_path)
            


# # %%
