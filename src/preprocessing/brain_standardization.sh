#!/usr/bin/bash


## INIT
while getopts t:u:o: flag
do
    case "${flag}" in
        t) t1_mod=${OPTARG};;
        u) t2_mod=${OPTARG};;
        o) output_dir=${OPTARG};;
    esac
done

source ~/.bashrc
mkdir $output_dir/warp
conda activate MRnorm


## VARS
MNItemplate=/home/tbarba/projects/MultiModalBrainSurvival/data/MR/templates/MNI152_T1_1mm_Brain.nii.gz


# REGISTRATION 
antsRegistrationSyNQuick.sh \
  -d 3 \
  -f $t1_mod \
  -m $t2_mod \
  -o $output_dir/warp/t2ToT1_

antsRegistrationSyNQuick.sh \
  -d 3 \
  -f $MNItemplate \
  -m $t1_mod \
  -o $output_dir/warp/t1ToTemplate_

antsApplyTransforms \
  -d 3 \
  -i $t1_mod \
  -o $output_dir/warp/t1DeformedToTemplate.nii.gz \
  -r $MNItemplate \
  -t $output_dir/warp/t1ToTemplate_1Warp.nii.gz \
  -t $output_dir/warp/t1ToTemplate_0GenericAffine.mat

antsApplyTransforms \
  -d 3 \
  -i $t2_mod \
  -o $output_dir/warp/t2DeformedToTemplate.nii.gz \
  -r $MNItemplate \
  -t $output_dir/warp/t1ToTemplate_1Warp.nii.gz \
  -t $output_dir/warp/t1ToTemplate_0GenericAffine.mat \
  -t $output_dir/warp/t2ToT1_0GenericAffine.mat


# INTENSITY
zscore-normalize $output_dir/warp/t1DeformedToTemplate.nii.gz -o $output_dir/normT1.nii.gz
zscore-normalize $output_dir/warp/t2DeformedToTemplate.nii.gz -o $output_dir/normT2.nii.gz
conda deactivate
# rm -rf $output_dir/warp