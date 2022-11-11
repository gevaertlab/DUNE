import os
from datetime import datetime, date
import humanfriendly

def main():
    ROOTDIR = "/home/tbarba/projects/MultiModalBrainSurvival/"
    os.chdir(ROOTDIR)

    batch_folder = "./config/batch_conf"
    files = [os.path.join(batch_folder, f) for f in os.listdir(batch_folder) if f.endswith("json")]

    for file in files[::-1]:

        print(f"Processing {file}...")
        command = f"CUDA_VISIBLE_DEVICES=0,1,2,3;\
        python 0_MRI/autoencoder/training.py\
        --config {file} \
        > outputs/{os.path.basename(file)}.log 2>&1"
        os.system(command)
        # print(command)
        print("Finished")   


if __name__ == "__main__":
    start = datetime.now()
    main()
    execution_time = humanfriendly.format_timespan(datetime.now() - start)
    print(f"\nFinished in {execution_time}.")