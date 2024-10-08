#!/bin/bash

#cd ../..

# custom config
DATA=/path/to/dataset
TRAINER=LinearProbingCLIP

DATASET=$1
SEED=$2
CUDA_VISIBLE_DEVICES=$3
BATCH=$4
EP=$5

CFG=mom_lr2e-3_B${BATCH}_ep${EP}
SHOTS=16


DIR=/path/to/output_dir/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/LFA/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi