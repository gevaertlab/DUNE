create_csvs:
	python aspects/1_HistoPathology/create_subset_csv.py

preproc_histo:
	python aspects/1_HistoPathology/1_WSI2Patches.py \
	--wsi_path "/home/tbarba/storage/Brain_pathology/WSIs" \
	--patch_path "/home/tbarba/storage/Brain_pathology/patches" \
	--mask_path "/home/tbarba/storage/Brain_pathology/masks"  \
	--patch_size 224 \
	--max_patches_per_slide 2000 \
	--num_process 10 \
	--dezoom_factor 1.0

train_histo:
	CUDA_VISIBLE_DEVICES=2 \
	python aspects/1_HistoPathology/2_HistoPath_train.py \
		--config "ExampleConfigs/config_ffpe_train.json" > logs/file_3epochs.log



# BRAIN AUTOENCODER

MODEL=UNet_5b_8f_UKfull
train_ae:
	CUDA_VISIBLE_DEVICES=2,3\
	python aspects/0_MRI/autoencoder/train_ae.py \
		--config "outputs/UNet/$(MODEL)/config/ae.json"

extract:
	CUDA_VISIBLE_DEVICES=0,3  \
	python aspects/0_MRI/autoencoder/feature_extraction.py -m $(MODEL) \
	--num_blocks 5 --init_feat 4

concat:
	python scripts/concat_features.py \
		--metadata data/metadata/whole_ukb_metadata.csv \
		--features_dir outputs/UNet/$(MODEL)/autoencoding/features

		# --metadata data/metadata/TCGA_survival_bins.csv \

# PREDICTIONS
univariate:
	Rscript aspects/0_MRI/predictions/univariate.r \
		--config /home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/$(MODEL)/config/univariate.json

predict:
	python aspects/0_MRI/predictions/train_pred.py \
		--config outputs/UNet/$(MODEL)/config/predict.json


# FULL PIPELINE
pipeline:
	python scripts/batch_ae.py
	python scripts/batch_extract.py
	python scripts/batch_pred.py