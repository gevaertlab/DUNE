import torch
from os.path import join
import torch.nn as nn
from tqdm import tqdm
import utils_ae as ut
import nibabel as nib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nifti_model = nib.load("/home/tbarba/projects/MultiModalBrainSurvival/data/MR/ADNI/processed/002_S_0295/normT1_crop.nii.gz")


def reconstruct_modality(net, model_path, fullLoader, device):

    net.eval()

    for imgs, case_folder in tqdm(fullLoader, colour="blue", desc=f"Modality Reconstruction : {model_path}"):

        imgs = imgs.to(device)
        case_folder = case_folder[0]

        with torch.no_grad():
            reconstructed, _, _ = net(imgs)

        reconstructed = reconstructed.squeeze().detach().cpu().numpy()
        reconstructed = reconstructed.transpose((2,1,0))

        reconstructed = nib.Nifti1Image(reconstructed, affine=nifti_model.affine)
        output_img = join(case_folder, "reconstructed.nii.gz")
        nib.save(reconstructed,output_img)


def main():
    config = ut.parse_arguments("ae")
    model_path = config['model_path']

    # Restoring model
    net, _ = ut.import_model(**config, device=device)
    net = nn.DataParallel(net)
    net = net.to(device)
    if config["other_model"]:
        print("Using other model : ", config["other_model"])
        dict_path = f"{config['other_model']}/exports/best_model.pt"
    else:
        dict_path = f"{model_path}/exports/best_model.pt"
    net.load_state_dict(torch.load(dict_path, map_location=torch.device('cpu')))

    # Restoring dataset
    fullLoader = ut.half_dataset(**config)

    # Extracting features
    reconstruct_modality(net, model_path, fullLoader, device)


if __name__ == "__main__":
    main()
