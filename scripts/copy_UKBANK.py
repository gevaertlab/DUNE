# %%
import os
import shutil
from os.path import join
from alive_progress import alive_bar


UK_BIOBANK = "/home/tbarba/storage/uk_biobank_2020"
OUTPUT_DIR = "/home/tbarba/projects/MultiModalBrainSurvival/data/data_fusion/MR/UKBIO"


if __name__ == "__main__":

    T1cases = [f[:12]
               for f in os.listdir(join(UK_BIOBANK, "T1")) if f.endswith("zip")]
    T2cases = [f[:12]
               for f in os.listdir(join(UK_BIOBANK, "T2")) if f.endswith("zip")]

    commoncases = [f for f in T1cases if f in T2cases]

    sequences = ["T1", "FLAIR"]

    with alive_bar(len(commoncases)) as bar:
        for case in commoncases:
            destination = f"{OUTPUT_DIR}/{case}"
            os.makedirs(destination, exist_ok=True)

            for seq in sequences:
                if seq == "T1":
                    img_path = f"{UK_BIOBANK}/T1/{case}2_2_0T1/T1/T1_brain.nii.gz"
                elif seq == "FLAIR":
                    img_path = f"{UK_BIOBANK}/T2/{case}3_2_0T2/T2_FLAIR/T2_FLAIR_brain.nii.gz"

                newname = f"{case}_{seq}.nii.gz"

                try:
                    shutil.copy(img_path, join(destination, newname))
                except FileNotFoundError:
                    if os.path.exists(destination):
                        shutil.rmtree(destination)

            bar()

    print("Finished.")
