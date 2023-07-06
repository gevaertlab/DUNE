# %% init
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cross_decomposition import CCA
# https://cmdlinetips.com/2020/12/canonical-correlation-analysis-in-python/
# https://medium.com/@pozdrawiamzuzanna/canonical-correlation-analysis-simple-explanation-and-python-example-a5b8e97648d2



emb_AE = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/test/UKB/exports/features/whole_brainAE.csv.gz"
emb_UNet = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/test/UKB/exports/features/whole_brainUNET.csv.gz"
emb_VAE3D = "/home/tbarba/projects/MultiModalBrainSurvival/outputs/test/UKB/exports/features/whole_brainVAE3D.csv.gz"


metadata = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/metadata/0-UKB_metadata_encoded.csv.gz"
radiomics = "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/metadata/0-WB_pyradiomics.csv.gz"


metadata_var = pd.read_csv(
    "/home/tbarba/projects/MultiModalBrainSurvival/data/MR/UKBIOBANK/metadata/0-variable_list.csv")
metadata_var = metadata_var[["var", "category"]].set_index("var")


emb_AE = pd.read_csv(emb_AE, index_col="eid")
emb_UNet = pd.read_csv(emb_UNet, index_col="eid")
emb_VAE3D = pd.read_csv(emb_VAE3D, index_col="eid")
metadata = pd.read_csv(metadata, index_col="eid")
radiomics = pd.read_csv(radiomics, index_col="eid")


# %%
emb_AE = emb_AE.sort_index()
emb_VAE3D = emb_VAE3D.sort_index()


radiomics = radiomics.sort_index()
metadata = metadata.loc[emb_AE.index].sort_index()
radiomics = radiomics.loc[emb_AE.index].sort_index()


emb_UNet = emb_UNet.sort_index()
emb_UNet = emb_UNet.loc[metadata.index]

# %%


def perform_CCA(dataset, model, n_comp=2):

    ca = CCA(n_components=n_comp)
    ca.fit(dataset, metadata)
    X_c, Y_c = ca.transform(dataset, metadata)

    ccX_df = pd.DataFrame(X_c, index=metadata.index)
    ccX_df.columns = [f"{model}_CC{i}" for i, _ in enumerate(ccX_df.columns)]

    merged = ccX_df.merge(metadata, left_index=True, right_index=True)

    corr_X_df = merged.corr(method='pearson')
    corr_X_df = corr_X_df.where(np.tril(np.ones(corr_X_df.shape)).astype(bool))

    corr_X_df = corr_X_df.iloc[n_comp:, :n_comp]

    return corr_X_df


# %%
# AE = perform_CCA(emb_AE, "AE",  n_comp=1)
# VAE3D = perform_CCA(emb_VAE3D, "VAE3D",  n_comp=1)
UNet = perform_CCA(emb_UNet, "UNet",  n_comp=1)
# radiomics = perform_CCA(radiomics, "radiomics", n_comp=1)



# %%
fused = pd.concat([AE, VAE3D, radiomics], axis=1).sort_index()
fused = pd.merge(fused, metadata_var, left_index=True, right_index=True)

custom_dict = ["mri", 'meta', 'global', 'lifestyle', 'diet', 'toxics', 'cvrf',  'vascular', 'psy', 'function', 'genetic', 'conditions']
custom_dict = {k:o for o,k in enumerate(custom_dict)}
fused = fused.sort_values("category", key=lambda x: x.map(custom_dict))
categories = fused.pop("category")

fused = np.square(fused)



n_colors = len(custom_dict)
colours = cm.gnuplot(np.linspace(0, 1, n_colors))

lut = dict(zip(categories.unique(), colours))
row_colors = categories.map(lut)


sns.clustermap(fused,
               cmap="mako",
               norm=LogNorm(),
               annot=False, fmt='.1g',
               yticklabels=False,
               tree_kws={"linewidths": 0.}, row_cluster=False,
               row_colors=row_colors)

handles = [Patch(facecolor=lut[name]) for name in lut]

plt.legend(handles, lut, title='Species',
           bbox_transform=plt.gcf().transFigure,
           bbox_to_anchor=(.05, .5), loc='center left')


# %%
