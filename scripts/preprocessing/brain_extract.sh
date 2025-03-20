# brain_extract.sh
#!/usr/bin/bash

while getopts i:o: flag
do
    case "${flag}" in
        i) input=${OPTARG};;
        o) output=${OPTARG};;
    esac
done

brainextr_template=/Users/tom/Documents/temp/data/templates/OASIS/T_template0.nii.gz
brainextr_prob_mask=/Users/tom/Documents/temp/data/templates/OASIS/T_template0_BrainCerebellumProbabilityMask.nii.gz
brainextr_registration_mask=/Users/tom/Documents/temp/data/templates/OASIS/T_template0_BrainCerebellumRegistrationMask.nii.gz

# BRAIN EXTRACTION
antsBrainExtraction.sh -d 3 \
    -a "$input" \
    -e "$brainextr_template" \
    -m "$brainextr_prob_mask" \
    -f "$brainextr_registration_mask" \
    -o "$output"

# Rename the output to match expected name
mv "${output}BrainExtractionBrain.nii.gz" "$output"