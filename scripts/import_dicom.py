# %%
import os
from os.path import join, isdir
import pydicom
import numpy as np
from alive_progress import alive_bar
import pandas as pd

DATA_FOLDER = "/srv/gevaertlab/data/radiology/MRI_Brain_Age"
OUTPUT_FOLDER = "/home/tbarba/projects/MultiModalBrainSurvival/data/data_fusion/MR/MRI_Brain_Age"


cases = {f:join(DATA_FOLDER,f, "study","series") for f in os.listdir(DATA_FOLDER) if isdir(join(DATA_FOLDER, f))}

metadata_df = pd.DataFrame()

with alive_bar(len(cases)) as bar:
    for case, path in cases.items():
        
        T1, T2 = [], []
        for file in os.listdir(path):

            file = pydicom.filereader.dcmread(join(path,file))


            # Get metadata
            variables = ["PatientID", "PatientSex","PatientAge"]
            row = {var: [file[var][:]] for var in variables}
            metadata = pd.DataFrame.from_dict(row, orient="columns")
            metadata_df = pd.concat([metadata_df, metadata], ignore_index=True)


            # Get Images
            if "ax t1" in file.SeriesDescription.lower():
                T1.append(file.pixel_array)
            elif "ax t2" in file.SeriesDescription.lower():
                T2.append(file.pixel_array)


        try:
            T1 = np.stack(T1)    
            T2 = np.stack(T2)

            target_folder = join(OUTPUT_FOLDER, case)

            os.makedirs(target_folder, exist_ok=True)
            # np.save(join(target_folder, f"{case}_T1.npy"), T1)
            # np.save(join(target_folder, f"{case}_T2.npy"), T2)
        except ValueError:
            pass
        bar()
    
metadata_df["PatientAge"] = metadata_df["PatientAge"].str.extract('(\d+)').astype(int)
metadata_df.to_csv(OUTPUT_FOLDER + "/metadata.csv", index=False)

# %%


