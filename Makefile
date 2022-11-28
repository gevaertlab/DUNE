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

MODEL=UNet_6blocks_UK_Feat4_fulldataset
training:
	CUDA_VISIBLE_DEVICES=0,1 \
	python aspects/0_MRI/autoencoder/training.py \
		--config "outputs/UNet/$(MODEL)/config/ae.json" \
		> outputs/UNet/$(MODEL)/file.log 2>&1

extract_concat:
	CUDA_VISIBLE_DEVICES=2,3 \
	python aspects/0_MRI/autoencoder/feature_extraction.py -m $(MODEL) -b 6 ; \
	
	python scripts/concat_features.py \
		--metadata survival/whole_ukb_metadata.csv \
		--features_dir outputs/UNet/$(MODEL)/autoencoding/features \

univariate:
	Rscript aspects/0_MRI/predictions/univariate.r \
		--config /home/tbarba/projects/MultiModalBrainSurvival/outputs/UNet/$(MODEL)/config/univariate.json

predict:
	CUDA_VISIBLE_DEVICES=0 \
	python aspects/0_MRI/predictions/train_V3.py \
		--config outputs/UNet/$(MODEL)/config/height_pred.json \
		 > outputs/UNet/$(MODEL)/predictions/height/log.log 2>&1


