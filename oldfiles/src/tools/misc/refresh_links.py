import os
from glob import glob
import shutil

models = "/home/tbarba/projects/MultiModalBrainSurvival/outputs"
target = "/home/tbarba/projects/MultiModalBrainSurvival/configs"

if __name__ == "__main__":
    files = glob(models + "/*/*/config.cfg") + glob(models + "/*/*/*/config.cfg")
    
    try:
        shutil.rmtree(target)
        os.makedirs(target)
    except:
        pass


    for f in files:
        sl = f.split("/")[-2]
        sl = target + "/" + sl + ".cfg"
        os.symlink(f, sl)
