# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
# import wandb
# from tqdm import tqdm
from itertools import chain

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import (
    InfiniteDataLoader,
    FastDataLoader,
    DataParallelPassthrough,
)
from domainbed import model_selection
from domainbed.lib.query import Q

from datetime import datetime, timedelta

### wandb
# wandb.init(project="temp")  # debug name
# wandb.init(project="LFA-find-hparams")  # LFA, LFA-find-hparams

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str, default="RotatedMNIST")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--task",
        type=str,
        default="domain_generalization",
        help="domain_generalization | domain_adaptation",
    )
    parser.add_argument("--hparams", type=str, help="JSON-serialized hparams dict")
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 means "default hparams")',
    )
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and " "random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps. Default is dataset-dependent.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", type=str, default="train_output")
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--uda_holdout_fraction", type=float, default=0)
    parser.add_argument("--skip_model_save", action="store_true")
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    parser.add_argument("--use_caption", action="store_true")
    parser.add_argument("--feature_store_dir", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--find_hparams", type=int, default=0)
    parser.add_argument("--test_hparams", type=int, default=0)
    parser.add_argument("--load_pretrained", type=int, default=0)
    parser.add_argument("--activate_layer_lpclip", type=int, default=-1)

    # parser.add_argument('--clip_backbone', type=str, default="None")
    args = parser.parse_args()

    ### wandb
    if args.find_hparams == 1:
        args.steps = 3001   # 3000번의 스텝으로도 경향을 파악할 수 있다.
    #     if args.activate_layer_lpclip == -1:
    #         wandb.run.name = f"find_hparams-{args.output_dir.split('/')[-2]}-test{args.test_envs[0]}-trial_seed{args.trial_seed}-hparams_seed{args.hparams_seed}-step{args.steps}"
    #     else:
    #         wandb.run.name = f"find_hparams-{args.output_dir.split('/')[-2]}-test{args.test_envs[0]}-trial_seed{args.trial_seed}-hparams_seed{args.hparams_seed}-step{args.steps}-activate_layer_lpclip{args.activate_layer_lpclip}"
    # elif args.test_hparams == 1:
    #     if args.activate_layer_lpclip == -1:
    #         wandb.run.name = f"test_hparams-{args.output_dir.split('/')[-2]}-test{args.test_envs[0]}-trial_seed{args.trial_seed}-hparams_seed{args.hparams_seed}-step{args.steps}"
    #     else:
    #         wandb.run.name = f"test_hparams-{args.output_dir.split('/')[-2]}-test{args.test_envs[0]}-trial_seed{args.trial_seed}-hparams_seed{args.hparams_seed}-step{args.steps}-activate_layer_lpclip{args.activate_layer_lpclip}"
    # else:
    #     wandb.run.name = f"{args.output_dir.split('/')[-2]}-test{args.test_envs[0]}-trial_seed{args.trial_seed}-hparams_seed{args.hparams_seed}-step{args.steps}"
    # wandb.run.save()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))
    # sys.stderr = misc.Tee(os.path.join(args.output_dir, 'results.jsonl'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm,
            args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed),
        )
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    hparams["test_envs"] = [int(i) for i in args.test_envs]

    hparams["clip_transform"] = hparams["backbone"] == "clip"

    hparams["use_caption"] = args.use_caption

    hparams["dplclip_path"] = args.dplclip_path

    hparams["feature_store_dir"] = args.feature_store_dir

    hparams["rank"] = args.rank

    hparams["find_hparams"] = args.find_hparams

    hparams["test_hparams"] = args.test_hparams
    
    hparams["load_pretrained"] = args.load_pretrained
    
    hparams['dataset'] = args.dataset 

    hparams["activate_layer_lpclip"] = args.activate_layer_lpclip

    print("HParams:")
    for k, v in sorted(hparams.items()):
        print("\t{}: {}".format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    # NOTE: test_env의 것을 제외한 모든 out-split은 Validation에 사용되고 오직 test_env의 in-split만 Test에 사용된다.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(
            env,  # 0.2(test) / 0.8(train)
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )

        if env_i in args.test_envs:  # args.task != 'doamin_adaptation'이면 의미 X
            uda, in_ = misc.split_dataset(
                in_,  # 위 케이스의 경우, 앞에서 분할된 train은 그대로.
                int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i),
            )

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append(
            (in_, in_weights)
        )  # class별 분포를 고려하여 학습 데이터셋인 in_splits 생성, test_envs 포함
        out_splits.append(
            (out, out_weights)
        )  # class별 분포를 고려하여 평가 데이터셋인 out_splits 생성, test_envs 포함

        if len(uda):
            uda_splits.append((uda, uda_weights))

    train_loaders = [
        InfiniteDataLoader(  # 여기서 test_envs 제외
            dataset=env,
            weights=env_weights,
            batch_size=hparams["batch_size"],
            # batch_size=32,  # for 효율성 실험
            num_workers=dataset.N_WORKERS,
        )
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs
    ]

    uda_loaders = [
        InfiniteDataLoader(  # args.task != 'doamin_adaptation'이면 실행 X
            dataset=env,
            weights=env_weights,
            batch_size=hparams["batch_size"],
            # batch_size=32,  # for 효율성 실험
            num_workers=dataset.N_WORKERS,
        )
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs
    ]

    if (
        args.find_hparams
    ):  # Training-domain validation set (IIDAccuracySelectionMethod) 에서 동작한다.
        eval_loaders = [
            FastDataLoader(  # 모든 envs의 valid 데이터셋 생성, test 데이터셋 제외
                dataset=env, batch_size=64, num_workers=dataset.N_WORKERS
            )
            for env, _ in (out_splits + uda_splits)
        ]
        eval_weights = [None for _, weights in (out_splits + uda_splits)]
        eval_loader_names = ["env{}_out".format(i) for i in range(len(out_splits))]
        eval_loader_names += ["env{}_uda".format(i) for i in range(len(uda_splits))]
    elif (
        args.test_hparams
    ):  # Training-domain validation set (IIDAccuracySelectionMethod) 에서 동작한다.
        eval_loaders = [
            FastDataLoader(  # test_envs 데이터에 대한 test 데이터셋 생성
                dataset=env, batch_size=64, num_workers=dataset.N_WORKERS
            )
            for i, (env, _) in enumerate(in_splits)
            if i == args.test_envs[0]
        ]
        eval_loaders += [ # NOTE: 더 빠른 실험을 위해 validation을 제외하고 진행할 수 있음. 이 경우 마지막 step의 결과를 사용.
            FastDataLoader(  # best valid를 찾기 위해 모든 envs의 valid 데이터셋 추가
                dataset=env, batch_size=64, num_workers=dataset.N_WORKERS
            )
            for env, _ in (out_splits + uda_splits)
        ]
        eval_weights = [
            None for i, (_, weights) in enumerate(in_splits) if i == args.test_envs[0]
        ]
        eval_weights += [None for _, weights in (out_splits + uda_splits)]  # NOTE: 더 빠른 실험을 위해 validation을 제외하고 진행할 수 있음.
        eval_loader_names = ["env{}_in".format(args.test_envs[0])]
        eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]    # NOTE: 더 빠른 실험을 위해 validation을 제외하고 진행할 수 있음.
        eval_loader_names += ["env{}_uda".format(i) for i in range(len(uda_splits))]    # NOTE: 더 빠른 실험을 위해 validation을 제외하고 진행할 수 있음.
    else:
        eval_loaders = [
            FastDataLoader(  # 모든 envs의 valid+test 데이터셋 생성, 비효율적
                dataset=env, batch_size=64, num_workers=dataset.N_WORKERS
            )
            for env, _ in (in_splits + out_splits + uda_splits)
        ]
        eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
        eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
        eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
        eval_loader_names += ["env{}_uda".format(i) for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(args.test_envs),
        hparams,
    )

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if hasattr(algorithm, "network"):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams["batch_size"] for env, _ in in_splits])

    if args.algorithm in ["WordCLIP", "ZSCLIP"]:
        print('Setting step to 1...')
        args.steps = 1  # do not learn anything.

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    args.skip_model_save = False  # True

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict(),
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    now = datetime.now() + timedelta(hours=9)
    train_start = now
    print(
        "LOG:",
        f"start train. time is \t{now.strftime('%Y-%m-%d %H:%M:%S')}.",
    )
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [
            (x.to(device), y.to(device), path, label)
            for (x, y), path, label in next(train_minibatches_iterator)
        ]
        if args.task == "domain_adaptation":
            uda_device = [
                (x.to(device), path, label)
                for (x, _), path, label in next(uda_minibatches_iterator)
            ]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        # if step % 100 == 0:
        #     print(f"loss: {step_vals['loss']:2f}\t {step}/{n_steps-1}")
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            now = datetime.now() + timedelta(hours=9)
            train_time = now - train_start
            print(
                "LOG:",
                f"stop train at step {step}. time is \t{now.strftime('%Y-%m-%d %H:%M:%S')}. train time is \t{str(train_time)}.",
            )        
            # if step > 0: break # for 효율성 실험
            now = datetime.now() + timedelta(hours=9)
            eval_start = now
            print(
                "LOG:",
                f"start evaluation. time is \t{now.strftime('%Y-%m-%d %H:%M:%S')}.",
            )
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name + "_acc"] = acc
                # wandb.log({name+'_acc' : acc})  # wandb
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)
            
            now = datetime.now() + timedelta(hours=9)
            eval_time = now - eval_start
            print(
                "LOG:",
                f"stop evaluation. time is \t{now.strftime('%Y-%m-%d %H:%M:%S')}. evaluation time is \t{str(eval_time)}.",
            )

            results.update({"hparams": hparams, "args": vars(args)})

            # wandb.config.update(hparams)  # wandb

            epochs_path = os.path.join(args.output_dir, "results.jsonl")

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            records = []
            with open(epochs_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
            records = Q(records)
            scores = records.map(model_selection.IIDAccuracySelectionMethod._step_acc)
            if scores[-1] == scores.argmax("val_acc"):
                save_checkpoint("IID_best.pkl")
                algorithm.to(device)

            if args.save_model_every_checkpoint:
                save_checkpoint(f"model_step{step}.pkl")
                
            train_start = now
    save_checkpoint("model.pkl")

    with open(os.path.join(args.output_dir, "done"), "w") as f:
        f.write("done")