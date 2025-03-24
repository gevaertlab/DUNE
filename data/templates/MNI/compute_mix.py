import os
import nibabel as nib


template_dir = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/templates/MNI/"
T1 = template_dir + "MNI152_T1_1mm_Brain.nii.gz"
T2 = template_dir + "MNI152_T2_1mm_Brain.nii.gz"
mix_path = template_dir + "MNI152_mixed_template.nii.gz"
mix_path_zscore = template_dir + "MNI152_mixed_template_zscore.nii.gz"
SLICE  = 160

T1 = nib.load(T1)
T1_array = T1.get_fdata()

T2 = nib.load(T2)
T2_array = T2.get_fdata()

mix = T1_array * T2_array

mix = nib.Nifti1Image(mix, affine=T1.affine)


mix.slicer[:,:,:SLICE].to_filename(mix_path)


os.system(f"zscore-normalize {mix_path} -o {mix_path_zscore}")
