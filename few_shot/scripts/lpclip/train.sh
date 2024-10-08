#!/bin/bash

#cd ../..

# custom config
DATA=/path/to/dataset
TRAINER=LinearProbingCLIP

DATASET=$1
SEED=$2
SHOTS=$3
CUDA_VISIBLE_DEVICES=$4
BATCH=$5
EP=$6

CFG=mom_lr2e-3_B${BATCH}_ep${EP} #mom_lr2e-3_B256_ep40  # rn50, rn101, vit_b32 or vit_b16

DIR=/path/to/output_dir/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/LFA/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS}