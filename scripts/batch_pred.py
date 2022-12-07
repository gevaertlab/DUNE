import os
from datetime import datetime, date
import humanfriendly

def main():
    ROOTDIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOTDIR)

    list_of_dependents = [
        "age","sex","alcohol"
        ]


    for idx, model in enumerate(list_of_models):

        print(f"Training {model}, ({idx+1}/{len(list_of_models)})")
        cmd = f"CUDA_VISIBLE_DEVICES=0,3 \
	python aspects/0_MRI/autoencoder/train_ae.py \
    --config 'outputs/UNet/{model}/config/ae.json'"

        os.system(cmd)





if __name__ == "__main__":
    main()
    print(f"\Finished.")
