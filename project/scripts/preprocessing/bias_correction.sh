# bias_correction.sh
#!/usr/bin/bash

while getopts i:o: flag
do
    case "${flag}" in
        i) input=${OPTARG};;
        o) output=${OPTARG};;
    esac
done

# BIAS CORRECTION
N4BiasFieldCorrection \
  -d 3 \
  --shrink-factor 3 \
  --bspline-fitting [ 300 ] \
  --convergence [ 50x50x30x20 ] \
  --input-image "$input" \
  --output "$output"