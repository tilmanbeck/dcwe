#!/bin/bash
#
#SBATCH --job-name=sandy-dcwe
#SBATCH --output=/ukp-storage-1/beck/slurm_output/sandy_dcwe
#SBATCH --mail-user=beck@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=turtok

DATANAME=sandy
DATAFILE=/ukp-storage-1/beck/Repositories/temporal-adaptation/datasets/stowe-2018/stowe-2018-labeled-all.csv

EPOCHS=3
BATCH_SIZE=64
WARMUP=0.0
WEIGHT_DECAY=0.001
LR=0.0001
TIMERANGE=bin

source /ukp-storage-1/beck/Repositories/dcwe/venv/bin/activate
module purge
module load cuda/11.0

# sandy time_stratified_partition
for seed in {0..4}
do
#	python /ukp-storage-1/beck/Repositories/dcwe/src/main_temporal_classification.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/sandy/time_stratified --lm_model bert-base-cased --partition time_stratified_partition --seed $seed --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR
	python /ukp-storage-1/beck/Repositories/dcwe/src/main_temporal_classification.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/sandy/timebin/time_stratified --lm_model bert-base-cased --partition time_stratified_partition --seed $seed --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR --timerange $TIMERANGE
done

# sandy controlled_partition
for seed in {0..4}
do
#	python /ukp-storage-1/beck/Repositories/dcwe/src/main_temporal_classification.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/sandy/controlled --lm_model bert-base-cased --partition controlled_partition --seed $seed --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR
	python /ukp-storage-1/beck/Repositories/dcwe/src/main_temporal_classification.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/sandy/timebin/controlled --lm_model bert-base-cased --partition controlled_partition --seed $seed --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR --timerange $TIMERANGE
done
