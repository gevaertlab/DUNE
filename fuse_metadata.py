# %%
import pandas as pd


a = pd.read_csv(
    "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/all_cohorts/metadata/all_cohorts_single.csv.gz")

a = a.query("dataset !='ADNI'")

a["AD"] = 0
# %%

ADNI = pd.read_csv("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/ADNI/metadata/0-ADNI_metadata_encoded.csv"
                   )[["eid", "Group", "Sex", "Age", "dx"]]


ADNI["mod"] = "normT1_crop"
ADNI["id"] = ADNI["eid"]
ADNI["eid"] = ADNI["eid"].astype(str) + "__" + ADNI["mod"].astype(str)
ADNI["mod"] = "T1"
ADNI["AD"] = ADNI["dx"]
ADNI["Cancer"] = 0
ADNI["dataset"] = "ADNI"

ADNI = ADNI.drop(["dx", "Group", "Age"], axis=1)


final_df = pd.concat([a, ADNI])




# %%


TCGA = pd.read_csv("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/TCGA/metadata/0-TCGA_metadata_encoded.csv")

TCGA = TCGA[["eid","Gender"]]
TCGA = TCGA.rename({"eid":"id", "Gender":"Sex"}, axis=1)
# %%


final_df = final_df.merge(TCGA, left_on="id", right_on="id", how="left")

final_df["Sex_x"].update(final_df.pop("Sex_y"))
final_df = final_df.rename({"Sex_x":"Sex"}, axis=1)


final_df.to_csv("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/all_cohorts/metadata/all_cohorts_single.csv.gz", index=False)

# %%

fused_mods = final_df.copy()
fused_mods = fused_mods.drop(["eid","mod"], axis=1)
fused_mods = fused_mods.drop_duplicates(subset="id")
fused_mods = fused_mods.rename({"id":"eid"}, axis=1)
fused_mods = fused_mods.set_index("eid")
fused_mods.to_csv("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/all_cohorts/metadata/all_cohorts.csv.gz", index=True)

# %%
