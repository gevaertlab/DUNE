import os
from datetime import datetime, date
import re

def main():
    ROOTDIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOTDIR)

    list_of_models = [

        "UNet_5b_4f_UKfull",
        "UNet_5b_8f_UKfull",
        "UNet_6b_4f_UKfull",
        "UNet_6b_8f_UKfull"
        ]


    for idx, model in enumerate(list_of_models):

        blocks, features = re.findall(r'\d+', model) 

        print(f"Extracting features with {model}, ({idx+1}/{len(list_of_models)})")
        extract = f"CUDA_VISIBLE_DEVICES=2,3 \
                python aspects/0_MRI/autoencoder/feature_extraction.py -m {model} \
	                --num_blocks {blocks} --init_feat {features}"

        os.system(extract)

        concat = f"python scripts/concat_features.py \
		--metadata data/metadata/whole_ukb_metadata.csv \
		--features_dir outputs/UNet/{model}/autoencoding/features"
        os.system(concat)


if __name__ == "__main__":
    main()
    print(f"\nFinished.")
