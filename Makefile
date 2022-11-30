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

MODEL=UNet_6blocks_UKfull_Feat8
training:
	CUDA_VISIBLE_DEVICES=0,2 \
	python aspects/0_MRI/autoencoder/train_ae.py \
		--config "outputs/UNet/$(MODEL)/config/ae.json" \
		> outputs/UNet/$(MODEL)/file.log 2>&1

extract_concat:
	CUDA_VISIBLE_DEVICES=2,3 \
	python aspects/0_MRI/autoencoder/feature_extraction.py -m $(MODEL) -b 6 ; \
	
	python scripts/concat_features.py \
		--metadata survival/whole_ukb_metadata.csv \
		--features_dir outputs/UNet/$(MODEL)/autoencoding/features \




# PREDICTIONS
MODEL=UNet_6blocks_UKfull_Feat4
univariate:
	Rscript aspects/0_MRI/predictions/univariate.r \
		--config /home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/$(MODEL)/config/univariate.json

predict:
	CUDA_VISIBLE_DEVICES=3 \
	python aspects/0_MRI/predictions/train_pred.py \
		--config outputs/UNet/$(MODEL)/config/multivariate.json \
		> outputs/UNet/$(MODEL)/predictions/multi.log 2>&1


