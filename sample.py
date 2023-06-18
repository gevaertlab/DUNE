import tqdm, shutil, random, os


datasets = ["UKBIOBANK","UCSF","UPENN"]
target = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/selection3/processed"
for dataset in datasets:
    root = f"/home/tbarba/projects/MultiModalBrainSurvival/data/MR/{dataset}/processed"
    folders = os.listdir(root)

    k = min(len(folders), 400)

    choices = random.sample(os.listdir(root), k=k)
    for c in tqdm.tqdm(choices):
        shutil.copytree(os.path.join(root, c), os.path.join(target, c))
