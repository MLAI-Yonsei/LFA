## Installation
### Install Dassl library on `few_shot/` directory
```sh
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

## Setting the environment
### Enter the path on script files(.sh)
```sh
# scripts
# ㄴmodel_name
#   ㄴtrain.sh, test.sh, ...

DATA=/path/to/datasets
DIR=/path/to/output_dir/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
```
### Enter the path on trainer files(.py)
```py
# trainers
# ㄴlfa_clip_adapter.py, lfa_coop.py, lfa_maple.py

def load_pretrained_model(cfg, model_name):
    ...
    model.load_model(
        f"/path/to/output_dir/dataset/trainer/config/seed",
        epoch=40) # NOTE: enter the path where the pretrained model is located.
    ...
```
NOTE: do not change the values of `dataset`, `trainer`, `config`, and `seed`.

## Training and Evaluation
### Training
```sh
# example: bash train_regular.sh 1 0 lfa
bash train_regular.sh SEED CUDA_VISIBLE_DEVICES TRAINER
```
NOTE: change SEED, CUDA_VISIBLE_DEVICES, and TRAINER.
### Evaluation
```sh
# example: bash test_regular.sh 1 0 lfa
bash test_regular.sh SEED CUDA_VISIBLE_DEVICES TRAINER
```
NOTE: change SEED, CUDA_VISIBLE_DEVICES, and TRAINER.
