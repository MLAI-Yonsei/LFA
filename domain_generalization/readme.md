## DG experiment scripts.
### Experimenting on the given environment.
> 💡 This paragraph has been borrowed directly from [DPLCLIP's](https://github.com/shogi880/DPLCLIP/blob/master/README.md) official repository.
```sh    
python -m domainbed.scripts.train \
       --data_dir /my/datasets/path \
       --output_dir /my/output_dir/path \
       --algorithm ALGORITHM \
       --dataset DATASET \
       --hparams "{\"clip_backbone\": \"ViT-B/16\"}" 
```
Note: change ` --algorithms ALGORITHM --dataset DATASET` for different experiments.

### Experimenting on the random environments.
```sh
python -m domainbed.scripts.sweep launch \
       --data_dir /my/datasets/path \
       --output_dir /my/output_dir/path \
       --algorithms ALGORITHM \
       --datasets DATASET \
       --command_launcher local \
       --single_test_envs \
       --n_trials 3 \
       --n_hparams 5 \
       --find_hparams 1 or 0 (default: 0) \
       --test_hparams 1 or 0 (default: 0) \ 
       --load_pretrained 1 or 0 (default: 0) \
       --activate_layer_lpclip 0~11 (default: -1) \
       --hparams "{\"clip_backbone\": \"ViT-B/16\"}"
	
```
Note: if you want to use multi-gpu, change ` --command_launcher local` to ` --command_launcher multi_gpu`.
- you can use `--find_hparams 0 --test_hparams 0` for the original Domainbed benchmark's training and validation. However, this method is inefficient, so we suggest the following find-test methodology:
  1. set `--find_hparams 1`. the train step is fixed to 3001, and only the validation set is used to efficiently find the best `hparams_seed` and `trial_seed`.
  2. set `--test_hparams 1`. this will use the test set of `test_envs` and all validation sets to find the best validation score (IIDAccuracySelectionMethod).
  - while the logic is the same, this approach is more efficient.

### Print performance based on model selection methods.
```sh
python -m domainbed.scripts.collect_results --input_dir /my/output_dir/path/
```
