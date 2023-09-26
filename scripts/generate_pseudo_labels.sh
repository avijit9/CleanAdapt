#!/bin/bash
#SBATCH -A avijit.d
#SBATCH -n 37
#SBATCH -c 1
#SBATCH --gres=gpu:4
#SBATCH --mem=110000
#SBATCH --time=8-00:00:00


set -e 

eval "$(conda shell.bash hook)"
conda activate domain-adaptation

SEED=1


source_dataset=$1
target_dataset=$2
modality=$3
num_classes=$4

CUDA_VISIBLE_DEVICES=0 python3 generate_pseudo_labels.py \
--source_dataset $source_dataset --target_dataset $target_dataset \
--modality $modality \
--num_classes $num_classes \
--data_path ./data \
--pretrained ./models/ \
--split_path ./splits --save_dir ./results \
--num_workers 10 \
--seed $SEED \
--pretrained_weight_path ./results/"$source_dataset"_"$target_dataset"_source_only_$modality/checkpoints/checkpoint.pth.tar \

# clean all __pycache__ and .pyc files
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf