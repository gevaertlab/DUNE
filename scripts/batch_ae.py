import os
from datetime import datetime, date
import humanfriendly

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

        print(f"Training {model}, ({idx+1}/{len(list_of_models)})")
        cmd = f"CUDA_VISIBLE_DEVICES=2,3 \
	python aspects/0_MRI/autoencoder/train_ae.py \
    --config 'outputs/UNet/{model}/config/ae.json'"

        os.system(cmd)





if __name__ == "__main__":
    main()
    print(f"\nFinished.")
