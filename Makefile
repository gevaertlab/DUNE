# BRAIN AUTOENCODER
MODEL=VAE/VAE_UCSF_segm
MOD=3
UNET=0
CUDA=0,1
BLOCKS=6
FEATS=4

#
#TRAIN AE
train_ae:
	CUDA_VISIBLE_DEVICES=$(CUDA) \
	python src/autoencoder/train_ae.py \
		--model_path outputs/$(MODEL)

extract_vae_features:
	CUDA_VISIBLE_DEVICES=$(CUDA)  \
	python src/autoencoder/feature_extraction.py -m $(MODEL) \
	--num_blocks $(BLOCKS) --init_feat $(FEATS) --unet $(UNET) --num_mod $(MOD)

extract_radiomics:
	python src/tools/format_datasets/metadata/extract_radiomics.py \
		--model_path outputs/$(MODEL)

combine_feature_and_radiomics:
	python src/tools/format_datasets/metadata/combine_radiomics.py \
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
	python src/predictions/multivariate.py \
		--model_path outputs/$(MODEL)


predict:
	python src/predictions/train_pred.py \
		--task classification \
		--variable grade \
		--config outputs/$(MODEL)/config/predict.json


#
# SYNC WITH SHERLOCK
make sync:
	rsync -aP \
		/home/tbarba/projects/MultiModalBrainSurvival \
		--exclude="data" \
		--exclude="bidon" \
		--exclude="results" \
		--exclude=".git" \
		--exclude="pyrightconfig.json" \
		dtn.sherlock.stanford.edu:/home/users/tbarba/projects/

	rsync -aP \
		/home/tbarba/projects/MultiModalBrainSurvival/data/outputs \
		dtn.sherlock.stanford.edu:/home/users/tbarba/storage/data_fusion

