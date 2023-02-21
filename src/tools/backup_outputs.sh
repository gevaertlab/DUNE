cd /home/tbarba/projects/MultiModalBrainSurvival/


now=$(date +"%Y-%m-%d")
tar_file="$now-outputs.tar.gz"
tar -zcvf "$tar_file" data/MR/outputs

rsync -avrP "$tar_file" dtn.sherlock.stanford.edu:/oak/stanford/groups/ogevaert/data/Brain_VAE/
rm "$tar_file"
