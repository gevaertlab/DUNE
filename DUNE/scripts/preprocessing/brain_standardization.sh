# brain_standardization.sh
#!/usr/bin/bash

while getopts t:u:o: flag
do
    case "${flag}" in
        t) t1_mod=${OPTARG};;
        u) t2_mod=${OPTARG};;
        o) output_dir=${OPTARG};;
    esac
done

MNItemplate=/Users/tom/Documents/temp/data/templates/MNI/MNI152_T1_1mm_Brain.nii.gz

# Create warp directory
mkdir -p "$output_dir/warp"

# REGISTRATION 
antsRegistrationSyNQuick.sh \
  -d 3 \
  -f "$MNItemplate" \
  -m "$t1_mod" \
  -o "$output_dir/warp/t1ToTemplate_"

antsApplyTransforms \
  -d 3 \
  -i "$t1_mod" \
  -o "$output_dir/warp/t1DeformedToTemplate.nii.gz" \
  -r "$MNItemplate" \
  -t "$output_dir/warp/t1ToTemplate_1Warp.nii.gz" \
  -t "$output_dir/warp/t1ToTemplate_0GenericAffine.mat"

# INTENSITY
zscore-normalize "$output_dir/warp/t1DeformedToTemplate.nii.gz" -o "$output_dir/normT1.nii.gz"