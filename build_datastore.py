#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain
from typing import final

import torch
from omegaconf import DictConfig

import numpy as np
import pandas as pd
import os
import sys
from os import path

sys.path.append(path.join(path.dirname( path.abspath(__file__) ), "fairseq"))

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
# Get a logger with specified name
logger = logging.getLogger("fairseq_cli.build_knn_data_store")

def add_custom_arguments(parser):
            
    parser.add_argument('--dataset', type=str, default='BindingDB_IC50',
                        help='Use which dataset and model to build datastore')

    parser.add_argument('--datastore-path', type=str, default='tmp',
                        help='Where to save the datastore')

    return parser

def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    ################################
    
    #############################################################################
    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []

        id_tensor_list = []
        cls_0_tensor_list = []
        cls_1_tensor_list = []
        target_tensor_list = []
        src_tokens_0_tensor_list = []
        src_tokens_1_tensor_list = []

        # Iterate over the 'subset' dataset
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            # _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            model.eval()
            with torch.no_grad():
                _sample_size, log_output = criterion(model, sample)
           
            id_tensor_list.append(sample['id'].detach().cpu().numpy())
            cls_0_tensor_list.append(log_output['cls_0'].detach().cpu().numpy())
            cls_1_tensor_list.append(log_output['cls_1'].detach().cpu().numpy())
            target_tensor_list.append(log_output['target'].detach().cpu().numpy())
            # 利用 src_tokens 来去重
            src_tokens_0_tensor_list.append(log_output['src_tokens_0'].detach().cpu().numpy())
            src_tokens_1_tensor_list.append(log_output['src_tokens_1'].detach().cpu().numpy())

            # 原本的写法，在 reduce metric 里作存储会导致显存不断增大，因为每次都要存储id cls_0 cls_1 target，特别是高维tensor占内存较大
            # log_output_tmp = {'id': sample['id'], 'cls_0': log_output['cls_0'], 'cls_1': log_output['cls_1'], 'target': log_output['target'], 'sample_size': log_output['sample_size'], 'ntokens': log_output['ntokens'], 'nsentences': log_output['nsentences']}
            log_output_tmp = {'sample_size': log_output['sample_size'], 'ntokens': log_output['ntokens'], 'nsentences': log_output['nsentences']}
            progress.log(log_output_tmp, step=i)
            log_outputs.append(log_output_tmp)


        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)

        id_list, cls_0_list, cls_1_list, target_list, src_tokens_0_list, src_tokens_1_list = [], [], [], [], [], []
        for i in id_tensor_list:
            if i.ndim == 1:
                for j in i:
                    id_list.append(j)
            else: # i.ndim == 0
                id_list.append(i)

        for i in cls_0_tensor_list:
            if i.ndim == 2:
                for j in i:
                    cls_0_list.append(j)
            else: # i.ndim == 1
                cls_0_list.append(i)

        for i in cls_1_tensor_list:
            if i.ndim == 2:
                for j in i:
                    cls_1_list.append(j)
            else: # i.ndim == 1
                cls_1_list.append(i)

        for i in target_tensor_list:
            if i.ndim == 1:
                for j in i:
                    target_list.append(j)
            else: # i.ndim == 0
                target_list.append(i)
            
        for i in src_tokens_0_tensor_list:
            if i.ndim == 2:
                for j in i:
                    # src_tokens_0_list.append(j)
                    src_tokens_0_list.append(np.pad(j, [0, 1050 - len(j)], 'constant', constant_values=(1, 1)))
            else: # i.ndim == 1
                # src_tokens_0_list.append(i)
                src_tokens_0_list.append(np.pad(i, [0, 1050 - len(i)], 'constant', constant_values=(1, 1)))

        for i in src_tokens_1_tensor_list:
            if i.ndim == 2:
                for j in i:
                    # src_tokens_1_list.append(j)
                    src_tokens_1_list.append(np.pad(j, [0, 1050 - len(j)], 'constant', constant_values=(1, 1)))
            else: # i.ndim == 1
                # src_tokens_1_list.append(i)
                src_tokens_1_list.append(np.pad(i, [0, 1050 - len(i)], 'constant', constant_values=(1, 1)))

        df = pd.DataFrame({'cls_0': cls_0_list, 'cls_1': cls_1_list, 'target': target_list, 'src_tokens_0': src_tokens_0_list, 'src_tokens_1': src_tokens_1_list}, index=np.array(id_list))
        # 调整 training set 本身的顺序（原本为 fairseq 随机循环 batch 随机的顺序）
        df.sort_index(inplace=True)

        cls_0_np = np.array(df['cls_0'].to_list()).astype('float32')
        cls_1_np = np.array(df['cls_1'].to_list()).astype('float32')
        # 报错： unhashable
        # df_unique_mol = df.drop_duplicates(subset=['src_tokens_0'])
        # df_unique_pro = df.drop_duplicates(subset=['src_tokens_1'])
        # df['src_tokens_0'].apply(tuple)
        # df['src_tokens_1'].apply(tuple)
        df_unique_mol = df[~df['src_tokens_0'].apply(tuple).duplicated()]
        df_unique_pro = df[~df['src_tokens_1'].apply(tuple).duplicated()]
        unique_cls_0_np = np.array(df_unique_mol['cls_0'].to_list()).astype('float32')
        unique_cls_1_np = np.array(df_unique_pro['cls_1'].to_list()).astype('float32')
        # unique_cls_0_np = np.array(list(set([tuple(t) for t in df['cls_0'].to_list()]))).astype('float32')
        # unique_cls_1_np = np.array(list(set([tuple(t) for t in df['cls_1'].to_list()]))).astype('float32')

        if not os.path.exists(cfg.criterion.datastore_path):
            os.makedirs(cfg.criterion.datastore_path)
        
        np.save(f'{cfg.criterion.datastore_path}/cls_0', cls_0_np)
        np.save(f'{cfg.criterion.datastore_path}/cls_1', cls_1_np)
        np.save(f'{cfg.criterion.datastore_path}/cls_0_unique_mol', unique_cls_0_np)
        np.save(f'{cfg.criterion.datastore_path}/cls_1_unique_pro', unique_cls_1_np)

        logger.info(f"Build datastore for {cfg.criterion.dataset} {cfg.dataset.valid_subset} set, size: {log_output['bsz']}\ntraining_set_size: {len(cls_0_np)} \nunique_molecule_num: {len(unique_cls_0_np)} \nunique_protein_num: {len(unique_cls_1_np)}")

def cli_main():
    parser = options.get_validation_parser()
    parser = add_custom_arguments(parser)
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_parser = add_custom_arguments(override_parser)
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()
