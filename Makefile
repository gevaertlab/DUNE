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
training:
	CUDA_VISIBLE_DEVICES=0,2,3 \
	python aspects/0_MRI/autoencoder/training.py \
		--config "outputs/UNet/testUnet/config/ae.json" \
		> outputs/UNet/testUnet/file.log 2>&1

feature_extraction:
	CUDA_VISIBLE_DEVICES=0,1,2 \
	python aspects/0_MRI/autoencoder/feature_extraction.py -m UNet_5blocks_TCGA_Feat4 

concat:
	python scripts/concat_features.py \
		--metadata survival/uk_metadata.csv \
		--features_dir outputs/UNet/UNet3D_5blocks_UK_Feat4/results/features \

univariate:
	Rscript aspects/0_MRI/predictions/univariate.r \
		--config /home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/UNet_5blocks_UK_Feat4/config/univariate.json

predict:
	CUDA_VISIBLE_DEVICES=2 \
	python aspects/0_MRI/predictions/train_V3.py \
		--config outputs/UNet/UNet_5blocks_UK_Feat4/config/age_pred.json \
		 > outputs/UNet/UNet_5blocks_UK_Feat4/surv.log 2>&1

