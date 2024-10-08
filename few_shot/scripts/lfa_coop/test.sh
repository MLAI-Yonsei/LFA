DATASET=$1
SEED=$2
SHOTS=$3
CUDA_VISIBLE_DEVICES=$4
BATCH=$5
EP=$6

# custom config
TRAINER=LFAonCoOp
CFG=mom_lr2e-3_B${BATCH}_ep${EP}
DATA=/path/to/datasets
DIR=/path/to/output_dir/evaluation/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
LOADEP=${EP}

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
    --model-dir /path/to/model_dir/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only
fi