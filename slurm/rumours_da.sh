#!/bin/bash
#
#SBATCH --job-name=rumours-da
#SBATCH --output=/ukp-storage-1/beck/slurm_output/rumours_da
#SBATCH --mail-user=beck@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=turtok

DATANAME=rumours
DATAFILE=/ukp-storage-1/beck/Repositories/temporal-adaptation/datasets/rumoureval/rumours-all.csv

EPOCHS=3
BATCH_SIZE=64
WARMUP=0.0
WEIGHT_DECAY=0.001
LR=0.0001
TIMERANGE=bin
LAMBDA=0.5

source /ukp-storage-1/beck/Repositories/dcwe/venv/bin/activate
module purge
module load cuda/11.0

# rumours time_stratified_partition
for seed in {0..4}
do
	python /ukp-storage-1/beck/Repositories/dcwe/src/main_temporal_adaptation.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/rumours/da/time_stratified --lm_model bert-base-cased --partition time_stratified_partition --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR --seed $seed --timerange $TIMERANGE --lambd $LAMBDA
done

# rumours controlled_partition
for seed in {0..4}
do
	python /ukp-storage-1/beck/Repositories/dcwe/src/main_temporal_adaptation.py --data_name $DATANAME --data_dir $DATAFILE --results_dir /ukp-storage-1/beck/Repositories/dcwe/results/rumours/da/controlled --lm_model bert-base-cased --partition controlled_partition --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --n_epochs $EPOCHS --lr $LR --seed $seed --timerange $TIMERANGE --lambd $LAMBDA
done