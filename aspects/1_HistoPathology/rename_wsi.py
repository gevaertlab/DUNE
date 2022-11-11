# %%

import os
import shutil
from tkinter import W
import pandas as pd


WSIs_path = "/home/tbarba/storage/Brain_pathology/WSIs/"

def extract_wsi_files(wsi_path):
    for subfolder in os.walk(wsi_path):
        subfolder = os.path.join(wsi_path, subfolder[0])

        try:
            svs_file = [f for f in os.listdir(subfolder) if f.endswith(".svs")][0]
        except IndexError:
            continue

        orig = os.path.join(subfolder, svs_file)
        destin = os.path.join(wsi_path, svs_file)
        shutil.move(orig, destin)
        shutil.rmtree(subfolder)
        

if __name__ == "__main__":
    extract_wsi_files(WSIs_path)
# %%
