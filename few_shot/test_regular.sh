SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=$3

###################### 16-Shot ######################
bash scripts/${TRAINER}/test.sh stanford_cars ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh oxford_flowers ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh caltech101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh dtd ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh fgvc_aircraft ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh oxford_pets ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh food101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh ucf101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh eurosat ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh sun397 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/test.sh imagenet ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
