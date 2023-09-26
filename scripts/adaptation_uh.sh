#!/bin/bash

#SBATCH -A avijit.d
#SBATCH -n 30
#SBATCH -c 1
#SBATCH --gres=gpu:4
#SBATCH --mem=110000
#SBATCH --time=4-00:00:00
#SBATCH --output=../slurms-output/adaptation_new_%j.out


set -e 

# eval "$(conda shell.bash hook)"
# conda activate domain-adaptation

mkdir -p /ssd_scratch/cvit/avijit
mkdir -p /ssd_scratch/cvit/avijit/results

SEED=1

source_dataset=$1
target_dataset=$2
modality=$3
r=$4
num_classes=12



python3 adaptation.py \
--source_dataset $source_dataset \
--target_dataset $target_dataset \
--modality $modality \
--adaptation_mode SLT \
--data_path ./data \
--num_epochs 100 --lr 0.01 \
--batch_size 48 --milestones 20 40 \
--pretrained ./models/ \
--num_workers 15 \
--split_path ./splits --save_dir ./results \
--pseudo_label_path ./results/"$source_dataset"_"$target_dataset"_"$modality"/pseudo_annotations.json \
--pretrained_weight_path ./results/"$source_dataset"_"$target_dataset"_source_only_"$modality"/checkpoints/checkpoint.pth.tar \
--seed $SEED \
--gpus 4 \
--r $r 




# rsync -avz /ssd_scratch/cvit/avijit/results/* avijit.d@ada:/share3/avijit.d/experiments/


# clean all __pycache__ and .pyc files
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

