# BRAIN AUTOENCODER
MODEL=VAE3D/VAE3D_UCSF_b10
CUDA=2,3
#
#TRAIN AE
train_ae:
	CUDA_VISIBLE_DEVICES=$(CUDA) \
	python src/autoencoder/train_ae.py -c $(MODEL)

extract_vae_features:
	CUDA_VISIBLE_DEVICES=$(CUDA)  \
	python src/autoencoder/feature_extraction.py -c $(MODEL)


extract_radiomics:
	python src/tools/format_datasets/metadata/extract_radiomics.py \
		--model_path outputs/$(MODEL)

combine_feature_and_radiomics:
	python src/tools/format_datasets/metadata/combine_radiomics.py \
	--model_path outputs/$(MODEL)

combine_brain_and_tumor_feat:
	python src/tools/format_datasets/metadata/combine_whole_brain_and_tumor_features.py \
	--model_path outputs/$(MODEL)

extract:
	make extract_vae_features
	make combine_feature_and_radiomics

#
# PREDICTIONS
univariate:
	Rscript src/predictions/univariate.r \
		--config outputs/$(MODEL)/config/univariate.json


multivariate:
	python src/autoencoder/multivariate.py -c $(MODEL)


predict:
	python src/predictions/train_pred.py \
		--task classification \
		--variable grade \
		--config outputs/$(MODEL)/config/predict.json


make links:
	python src/tools/misc/refresh_links.py