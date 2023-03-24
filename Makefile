# BRAIN AUTOENCODER
MODEL=finetuning/6b_4f_UCSF_segm2
MOD=3
UNET=0
CUDA=3,2


train_ae:
	CUDA_VISIBLE_DEVICES=$(CUDA) \
	python src/autoencoder/train_ae.py \
		--model_path outputs/UNet/$(MODEL)

extract_features:
	CUDA_VISIBLE_DEVICES=$(CUDA)  \
	python src/autoencoder/feature_extraction.py -m $(MODEL) \
	--num_blocks 6 --init_feat 4 --unet $(UNET) --num_mod $(MOD)


extract_radiomics:
	python src/tools/format_datasets/metadata/extract_radiomics.py \
		--model_path outputs/UNet/$(MODEL)

combine_feature_and_radiomics:
	python src/tools/format_datasets/metadata/combine_radiomics.py

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
