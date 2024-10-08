# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        results = (
            records.group("args.hparams_seed")
            .map(
                lambda _, run_records: (
                    self.run_acc(
                        run_records
                    ),  # trial 별 가장 valid 점수가 높게 나온 dict를 반환한다.
                    run_records,
                )
            )
            .filter(lambda x: x[0] is not None)
            .sorted(  # 반복 실험 중에서도 성능이 좋게 나온 순서대로 정렬한다.
                key=lambda x: x[0]["val_acc"]
            )[::-1]
        )
        print(
            f"dataset: {results[0][1][0]['args']['dataset']}",
            f"best_test_acc: {results[0][0]['test_acc']:.5f}",
            f"test_envs: {results[0][1][0]['args']['test_envs']}",
        )
        print("best_val_output_dir:", results[0][1][0]["args"]["output_dir"])
        if results[0][0]["test_acc"] == -1:  # args.find_hparams == True
            # for x in results:
            #     print("val_acc:", x[0]['val_acc'])
            #     print("hparams_seed:", x[1][0]['args']['hparams_seed'])
            #     print("trial_seed:", x[1][0]['args']['trial_seed'])
            #     print("seed:", x[1][0]['args']['seed'])     # hparams나 trial과 동시에 seed도 계속 달라진다.

            print("\n**Found best hparams_seed on best validation score!**")
            print(
                f"best_val_acc: {results[0][0]['val_acc']:.5f}",
                f"test_envs: {results[0][1][0]['args']['test_envs']}",
                f"hparams_seed: {results[0][1][0]['args']['hparams_seed']}",
                f"trial_seed: {results[0][1][1]['args']['trial_seed']}",
            )
            print("best_val_output_dir:", results[0][1][0]["args"]["output_dir"])

            output_dir = "/".join(
                results[0][1][0]["args"]["output_dir"].split("/")[:-1]
            )
            results_dir = os.path.join(
                output_dir, f"test_env_{results[0][1][0]['args']['test_envs'][0]}"
            )
            os.makedirs(results_dir, exist_ok=True)
            with open(
                os.path.join(
                    results_dir,
                    f"best_valid_model_paths_per_trial_{results[0][1][0]['args']['trial_seed']}.txt",
                ),
                "w",
            ) as f:
                f.write(
                    results[0][1][0]["args"]["output_dir"] + "\n"
                )  # default, use fisrt line only.
                f.write(f"best_val_acc: {results[0][0]['val_acc']:.5f}" + "\n")
                f.write(
                    f"hparams_seed: {results[0][1][0]['args']['hparams_seed']}" + "\n"
                )
                f.write(f"seed: {results[0][1][0]['args']['seed']}")
        return results

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]["test_acc"]
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record["args"]["test_envs"][0]
        val_env_keys = []
        for i in itertools.count():
            if f"env{i}_out_acc" not in record:
                break
            if i != test_env:
                val_env_keys.append(f"env{i}_out_acc")
        test_in_acc_key = "env{}_in_acc".format(test_env)
        # print({
        #     'val_acc': np.mean([record[key] for key in val_env_keys]),
        #     'test_acc': record[test_in_acc_key]
        # })
        return {
            "val_acc": np.mean([record[key] for key in val_env_keys]),
            "test_acc": record[test_in_acc_key] if test_in_acc_key in record else -1,
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        result = test_records.map(self._step_acc).argmax("val_acc")
        if result["val_acc"] != result["val_acc"]:  # NaN Check
            result = test_records.map(self._step_acc)[-1]
        print(result)
        return result

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None
