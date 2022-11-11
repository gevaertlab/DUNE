import os


ROOTDIR = "/home/tbarba/projects/MultiModalBrainSurvival/data/data_fusion/MR/UKBIOBANK"

folderlist = [ROOTDIR +"/"+ f for f in os.listdir(ROOTDIR) if os.path.isdir(ROOTDIR + "/" +f)][3:]

all_files = []
for folder in folderlist:
	all_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if "T1Gd" in f])

# print(all_files)
for file in all_files:
	os.rename(file, file.replace("T1Gd","T1"))