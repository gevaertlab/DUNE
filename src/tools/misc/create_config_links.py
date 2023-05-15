import os
from glob import glob


models = "/home/tbarba/projects/MultiModalBrainSurvival/outputs"
target = "/home/tbarba/projects/MultiModalBrainSurvival/configs"

if __name__ == "__main__":
    files = glob(models + "/*/*/config*")  

    for f in files:
        sl = f.split("/")[-2]
        sl = target + "/" + sl + ".cfg"
        try:
            os.symlink(f, sl)
        except FileExistsError:
            pass
