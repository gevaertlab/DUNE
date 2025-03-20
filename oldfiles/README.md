Brain Autoencoder Repo


Different datasets

    - UKB ~ 20_000 cases
    - UCSF ~ 500 cases
    - UPENN ~ 400 cases
    - TCGA ~ 160 cases

    - REMBRANDT = not preprocessed
    - STANFORD GBM = not preprocessed
    - SCHIZO = not preprocessed
    - ADNI ? Age ? PISA ? Austrlian bannk ?


Different model architectures :

    - UNet = UNet
    - UNet_crop = UNet cropped on tumor
    - AE = UNet no skip connections
    - AE_crop = UNet no skip connections cropped on tumor
    - VAE = UNet like VAE
    - VAE3D = Terry's model
    - ResNet ?


REMBRANDT
        133,
        175,
        148

   TCGA  "min_dims": [
        145,
        175,
        148
    ],


UKB     "min_dims": [
        145,
        175,
        148
    ],