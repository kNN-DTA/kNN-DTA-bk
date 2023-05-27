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

import faiss
import numpy as np
import pandas as pd

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
logger = logging.getLogger("fairseq_cli.validate")

def add_custom_arguments(parser):
    # parser.add_argument('--arch', default='roberta_dti_mlm_regress',
    #                     help='The model architecture')

    parser.add_argument('--T', type=float, metavar='T', default=1000,
                        help='Temperature T for softmax on paired cls embedding distance')

    parser.add_argument('--T-0', type=float, metavar='T', default=1000,
                        help='Temperature T for softmax on molecule cls embedding distance')
    
    parser.add_argument('--T-1', type=float, metavar='T', default=1000,
                        help='Temperature T for softmax on protein cls embedding distance')

    parser.add_argument('--k', type=int, metavar='k', default=16,
                        help='k nearest neighbors for paired cls embedding')

    parser.add_argument('--k-0', type=int, metavar='k', default=16,
                        help='k nearest neighbors for molecule cls embedding')
    
    parser.add_argument('--k-1', type=int, metavar='k', default=16,
                        help='k nearest neighbors for protein cls embedding')

    parser.add_argument('--l', type=float, metavar='l', default=0.8,
                        help='The prediction weight')

    parser.add_argument('--l-update', type=float, metavar='l', default=1,
                        help='The weight used to update prediction')
    
    parser.add_argument('--knn-embedding-weight-0', type=float, metavar='l', default=0.8,
                        help='The knn embedding weight') 

    parser.add_argument('--knn-embedding-weight-1', type=float, metavar='l', default=0.8,
                        help='The knn embedding weight')   

    parser.add_argument('--alpha', type=float, metavar='a', default=0.707,
                    help='Alpha for embedding-wise search')      

    parser.add_argument('--dataset', type=str, required=True, default='BindingDB_IC50',
                        help='Use which dataset and model')
    
    parser.add_argument('--datastore-path', type=str, required=True, default='tmp',
                        help='The datastore path, differ with models and datasets')
#################################################################################################
    parser.add_argument('--sim', type=str, default='L2', choices=['L2', 'cosine', 'attn', 'dot'],
                        help='The similarity metric for search. Note that --sim attn is used with use-attn-cal at the same time.')

    parser.add_argument('--label-use-attn-cal', action='store_true',
                        help='Use attention calculation when doing label-wise search')

    parser.add_argument('--embedding-use-attn-cal', action='store_true',
                        help='Use attention calculation when doing embedding-wise search')

    parser.add_argument('--label-use-mean-cal', action='store_true',
                    help='Use mean calculation when doing label-wise search')

    parser.add_argument('--embedding-use-mean-cal', action='store_true',
                        help='Use mean calculation when doing embedding-wise search')

    parser.add_argument('--prediction-mode', type=str, required=True, default='combine', choices=['label', 'embedding', 'combine'],
                    help='The prediction mode of this script')

    parser.add_argument('--result-file-path', type=str, default='tmp.tsv',
                        help='Where to save the result tsv file')

    return parser

def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    ################################
    T = cfg.criterion.T
    T_0 = cfg.criterion.T_0
    T_1 = cfg.criterion.T_1
    l = cfg.criterion.l
    l_update = cfg.criterion.l_update
    knn_embedding_weight_0 = cfg.criterion.knn_embedding_weight_0
    knn_embedding_weight_1 = cfg.criterion.knn_embedding_weight_1
    k = cfg.criterion.k
    k_0 = cfg.criterion.k_0
    k_1 = cfg.criterion.k_1
    alpha = cfg.criterion.alpha
    d = 768 * 2       # dimension
    res = faiss.StandardGpuResources()  # use a single GPU
    if cfg.criterion.prediction_mode == 'label' or cfg.criterion.prediction_mode == 'combine':
        cls_0_np = np.load(f"{cfg.criterion.datastore_path}/cls_0.npy")
        cls_1_np = np.load(f"{cfg.criterion.datastore_path}/cls_1.npy")
          
        train_label_list = [float(i.strip()) for i in open(f'{cfg.task.data}/label/train.label').readlines()]
        cls_datastore = np.c_[cls_0_np, cls_1_np]

        
        # build a flat (CPU) index
        if cfg.criterion.sim == "L2":
            index_flat = faiss.IndexFlatL2(d)
        elif cfg.criterion.sim == "cosine" or cfg.criterion.sim == "attn" or cfg.criterion.sim == "dot":
            index_flat = faiss.IndexFlatIP(d)

        # make it into a gpu index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        if cfg.criterion.sim == "cosine":
            faiss.normalize_L2(cls_datastore)

        gpu_index_flat.add(cls_datastore)         # add vectors to the index
    #############################################################################
    if cfg.criterion.prediction_mode == 'embedding' or cfg.criterion.prediction_mode == 'combine':   
        cls_tmp_0 = np.load(f"{cfg.criterion.datastore_path}/cls_0_unique_mol.npy")
        cls_tmp_1 = np.load(f"{cfg.criterion.datastore_path}/cls_1_unique_pro.npy")
        # res = faiss.StandardGpuResources()  # use a single GPU
        # build a flat (CPU) index
        index_flat_0 = faiss.IndexFlatL2(int(d/2))
        index_flat_1 = faiss.IndexFlatL2(int(d/2))
        # make it into a gpu index
        gpu_index_flat_0 = faiss.index_cpu_to_gpu(res, 0, index_flat_0)
        gpu_index_flat_0.add(cls_tmp_0)         # add vectors to the index
        gpu_index_flat_1 = faiss.index_cpu_to_gpu(res, 0, index_flat_1)
        gpu_index_flat_1.add(cls_tmp_1)         # add vectors to the index
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
        prediction_tensor_list = []
        target_tensor_list = []

        # Iterate over the 'subset' dataset
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            # _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            model.eval()
            with torch.no_grad():
                _sample_size, log_output = criterion(model, sample)
            # get the query paired [CLS]
            # if _sample_size > 1:
            #     concat_tmp = np.c_[log_output['cls_0'].detach().cpu().numpy(), log_output['cls_1'].detach().cpu().numpy()]
            # else:
            #     concat_tmp = np.r_[log_output['cls_0'].detach().cpu().numpy(), log_output['cls_1'].detach().cpu().numpy()][np.newaxis, :]
            concat_tmp = np.c_[log_output['cls_0'].detach().cpu().numpy(), log_output['cls_1'].detach().cpu().numpy()]
            if cfg.criterion.sim == "cosine":
                faiss.normalize_L2(concat_tmp)

             
            if cfg.criterion.prediction_mode == 'embedding' or cfg.criterion.prediction_mode == 'combine':
                D_0, I_0 = gpu_index_flat_0.search(np.ascontiguousarray(concat_tmp[..., :int(d/2)]), k_0)
                D_1, I_1 = gpu_index_flat_1.search(np.ascontiguousarray(concat_tmp[..., int(d/2):]), k_1)
                V_cls_0 = np.array([cls_tmp_0[j] for i in range(_sample_size) for j in I_0[i]]).reshape(_sample_size, k_0, int(d/2))
                V_cls_1 = np.array([cls_tmp_1[j] for i in range(_sample_size) for j in I_1[i]]).reshape(_sample_size, k_1, int(d/2))
                               
                V_cls_0 = torch.from_numpy(V_cls_0).cuda()
                V_cls_1 = torch.from_numpy(V_cls_1).cuda()

                D_0 = torch.tensor(D_0).cuda()
                D_1 = torch.tensor(D_1).cuda()

                if cfg.criterion.embedding_use_attn_cal:
                    W_0 = torch.softmax(D_0 / torch.sqrt(torch.tensor(int(d/2))) , dim=1)  
                    W_1 = torch.softmax(D_1 / torch.sqrt(torch.tensor(int(d/2))) , dim=1) 
         
                else:
                    if cfg.criterion.sim == 'L2' or cfg.criterion.sim == 'dot':
                        W_0 = torch.softmax(- D_0 / T_0, dim=1)
                        W_1 = torch.softmax(- D_1 / T_1, dim=1)
                    elif cfg.criterion.sim == 'cosine':
                        W_0 = torch.softmax(D_0 / T_0, dim=1)
                        W_1 = torch.softmax(D_1 / T_1, dim=1)
                if cfg.criterion.embedding_use_mean_cal:
                    knn_cls_0 = torch.mean(V_cls_0, dim=1)
                    knn_cls_1 = torch.mean(V_cls_1, dim=1)
                else:
                # Weighted sum
                    knn_cls_0 = torch.sum(W_0[:, :, None] * V_cls_0, dim=1)
                    knn_cls_1 = torch.sum(W_1[:, :, None] * V_cls_1, dim=1)
                
                with torch.no_grad():
                    # _, log_output_new_cls = criterion(model, sample, knn_cls_0=knn_cls_0, knn_cls_1=knn_cls_1, use_which_embedding='mol_pro', knn_embedding_weight_0=knn_embedding_weight_0, knn_embedding_weight_1=knn_embedding_weight_1, alpha=alpha)
                    logits_regress, _, _ = model(
                        src_tokens_0 = sample["net_input"]["src_tokens_0"],
                        src_tokens_1 = sample["net_input"]["src_tokens_1"],
                        knn_cls_0 = knn_cls_0,
                        knn_cls_1 = knn_cls_1,
                        use_which_embedding = 'mol_pro',
                        knn_embedding_weight_0= knn_embedding_weight_0,
                        knn_embedding_weight_1= knn_embedding_weight_1,
                        alpha=alpha,
                        features_only=True,
                        classification_head_name='sentence_classification_head',
                        use_only_mlp=True,
                        cls_0=log_output['cls_0'],
                        cls_1=log_output['cls_1'],
                    ) 

                # update origin log output
                # log_output['prediction'] = log_output_new_cls['prediction']
                log_output['prediction'] = l_update * logits_regress + (1 - l_update) * log_output['prediction']

            if cfg.criterion.prediction_mode == 'label' or cfg.criterion.prediction_mode == 'combine':
                D, I = gpu_index_flat.search(concat_tmp, k)
                # _sample_size may be different in last batch
                V = torch.tensor([train_label_list[j] for i in range(_sample_size) for j in I[i]]).reshape(_sample_size, k).cuda()
                # W = softmax(- D / T, axis=1)
                D = torch.tensor(D).cuda()
                if cfg.criterion.label_use_attn_cal:
                    W = torch.softmax(D / torch.sqrt(torch.tensor(d)) , dim=1)
                    
                else:
                    if cfg.criterion.sim == 'L2' or cfg.criterion.sim == 'dot':
                        W = torch.softmax(- D / T, dim=1)
                    elif cfg.criterion.sim == 'cosine':
                        W = torch.softmax(D / T, dim=1)
                if cfg.criterion.label_use_mean_cal:
                    knn_prediction = torch.mean(V, dim=1)
                else:
                    # knn_prediction = np.sum(W * V, axis=1)
                    knn_prediction = torch.sum(W * V, dim=1)

                final_prediction = l * log_output['prediction'].squeeze() + (1 - l) * knn_prediction

            if cfg.criterion.prediction_mode == 'embedding':
                final_prediction = log_output['prediction'].squeeze()

            target = log_output['target']
            log_output_tmp = {'final_prediction': final_prediction, 'target': target, 'sample_size': log_output['sample_size'], 'ntokens': log_output['ntokens'], 'nsentences': log_output['nsentences']}
            progress.log(log_output_tmp, step=i)
            log_outputs.append(log_output_tmp)

            id_tensor_list.append(sample['id'].detach().cpu().numpy())
            prediction_tensor_list.append(final_prediction.detach().cpu().numpy())
            target_tensor_list.append(target.detach().cpu().numpy())

            


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

        id_list, prediction_list, target_list = [], [], []
        for i in id_tensor_list:
            if i.ndim == 1:
                for j in i:
                    id_list.append(j)
            else: # i.ndim == 0
                id_list.append(i)

        for i in prediction_tensor_list:
            if i.ndim == 1:
                for j in i:
                    prediction_list.append(j)
            else: # i.ndim == 0
                prediction_list.append(i)

        for i in target_tensor_list:
            if i.ndim == 1:
                for j in i:
                    target_list.append(j)
            else: # i.ndim == 0
                target_list.append(i)

        df = pd.DataFrame({'prediction': prediction_list, 'target': target_list}, index=np.array(id_list))
        # 调整 training set 本身的顺序（原本为 fairseq 随机循环 batch 随机的顺序）
        df.sort_index(inplace=True)
        
        df.to_csv(f'{cfg.criterion.result_file_path}', index=False, sep='\t')

        logger.info(f"{cfg.dataset.valid_subset} on {cfg.criterion.dataset}, input size is {log_output['bsz']}\nT={T}, T_0={T_0}, T_1={T_1}, k={k}, k_0={k_0}, k_1={k_1}, l={l}, knn_embedding_weight_0={knn_embedding_weight_0}, knn_embedding_weight_1={knn_embedding_weight_1}, alpha={alpha}, prediction mode={cfg.criterion.prediction_mode}\nMSE={log_output['MSE']}\nRMSE={log_output['RMSE']}\nPC={log_output['Pearson']}\nC-index={log_output['C-index']}")

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
