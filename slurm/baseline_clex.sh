#!/bin/bash
#
#SBATCH --job-name=clex-baseline
#SBATCH --output=/ukp-storage-1/beck/slurm_output/clex_baseline
#SBATCH --mail-user=beck@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=turtok

DATANAME=clex
DATAFILE=/ukp-storage-1/beck/Repositories/temporal-adaptation/datasets/clex/clex-all.csv

EPOCHS=3
BATCH_SIZE=64
WARMUP=0.0
WEIGHT_DECAY=0.001
LR=0.0001

source /ukp-storage-1/beck/Repositories/dcwe/venv/bin/activate
module purge
module load cuda/11.0

# clex time_stratified_partition
for seed in {0..4}
do
	python /ukp-storage-1/beck/Repositories/dcwe/src/main_classification.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/baseline/clex/time_stratified --lm_model bert-base-cased --seed $seed --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR
done

# clex controlled_partition
for seed in {0..4}
do
	python /ukp-storage-1/beck/Repositories/dcwe/src/main_classification.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/baseline/clex/controlled --lm_model bert-base-cased --seed $seed --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR
done
