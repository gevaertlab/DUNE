# BRAIN AUTOENCODER
MODEL=finetuning/6b_4f_UPENN_segm
MOD=3
CUDA=0,1
train_ae:
	CUDA_VISIBLE_DEVICES=$(CUDA) \
	python src/autoencoder/train_ae.py \
		--config outputs/UNet/$(MODEL)/config/ae.json

extract:
	CUDA_VISIBLE_DEVICES=$(CUDA)  \
	python src/autoencoder/feature_extraction.py -m $(MODEL) \
	--num_blocks 6 --init_feat 4 --num_mod $(MOD)

concat:
	python src/tools/concat_features.py \
		--metadata data/metadata/REMBRANDT_metadata.csv \
		--features_dir outputs/UNet/$(MODEL)/autoencoding/features

		# --metadata data/metadata/TCGA_metadata.csv \

# PREDICTIONS
univariate:
	Rscript src/predictions/univariate.r \
		--config outputs/UNet/$(MODEL)/config/univariate.json

multivariate:
	python src/predictions/multivariate.py \
		--model_path outputs/UNet/$(MODEL)

predict:
	python src/predictions/train_pred.py \
		--task classification \
		--variable grade \
		--config outputs/UNet/$(MODEL)/config/predict.json

# BACKUPS OUTPUTS ON OAK
backup:
	bash src/tools/backup_outputs.sh

# FULL PIPELINE
pipeline:
	python src/tools/batch/batch_ae.py
	python src/tools/batch/batch_extract.py
	python src/tools/batch/batch_pred.py




# create_csvs:
# 	python aspects/1_HistoPathology/create_subset_csv.py

# preproc_histo:
# 	python aspects/1_HistoPathology/1_WSI2Patches.py \
# 	--wsi_path "/home/tbarba/storage/Brain_pathology/WSIs" \
# 	--patch_path "/home/tbarba/storage/Brain_pathology/patches" \
# 	--mask_path "/home/tbarba/storage/Brain_pathology/masks"  \
# 	--patch_size 224 \
# 	--max_patches_per_slide 2000 \
# 	--num_process 10 \
# 	--dezoom_factor 1.0

# train_histo:
# 	CUDA_VISIBLE_DEVICES=2 \
# 	python aspects/1_HistoPathology/2_HistoPath_train.py \
# 		--config "ExampleConfigs/config_ffpe_train.json" > logs/file_3epochs.log

