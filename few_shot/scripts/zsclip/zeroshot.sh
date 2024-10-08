#!/bin/bash

#cd ../..

# custom config
DATA=/path/to/dataset
TRAINER=ZeroshotCLIP

DATASET=$1
SEED=$2

CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16

DIR=/path/to/output_dir/${DATASET}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir ${DIR} \
--eval-only