# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import os

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion("dti_separate_knn_build_datastore")
class DTIRegressKNNBuildDatastore(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) and regression training.
    """

    def __init__(self, task, classification_head_name):
        super().__init__(task)
        self.classification_head_name = classification_head_name

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')


    def forward(self, model, sample, reduce=True, knn_cls_0=0, knn_cls_1=0, use_which_embedding='no', knn_embedding_weight_0=0.8, knn_embedding_weight_1=0.8, alpha=0.707):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        
        """

        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        # Recover origin input by src and tgt tokens
        # regression. Inputs are unmasked tokens recovered by src and tgt tokens
        # To do: process these cls tokens
        logits_regress, cls_0, cls_1 = model(
            src_tokens_0 = sample["net_input"]["src_tokens_0"],
            src_tokens_1 = sample["net_input"]["src_tokens_1"],
            knn_cls_0 = knn_cls_0,
            knn_cls_1 = knn_cls_1,
            use_which_embedding = use_which_embedding,
            knn_embedding_weight_0= knn_embedding_weight_0,
            knn_embedding_weight_1= knn_embedding_weight_1,
            alpha=alpha,
            features_only=True,
            classification_head_name=self.classification_head_name,
        ) 

        # targets = model.get_targets(sample, [logits]).view(-1)
        targets_regress = sample["target"].view(-1)
        
        sample_size_regress = targets_regress.numel()

        # if not self.regression_target:
        #     lprobs = F.log_softmax(logits_regress, dim=-1, dtype=torch.float32)
        #     loss_regress = F.nll_loss(lprobs, targets_regress, reduction="sum")
        # else:
        #     logits_regress = logits_regress.view(-1).float()
        #     targets_regress = targets_regress.float()
        #     loss_regress = F.mse_loss(logits_regress, targets_regress, reduction="sum")

        logging_output = {
            "prediction": logits_regress,
            "target": targets_regress,
            "cls_0": cls_0,
            "cls_1": cls_1,
            "src_tokens_0": sample["net_input"]["src_tokens_0"],
            "src_tokens_1": sample["net_input"]["src_tokens_1"],
            "ntokens": sample["ntokens_0"] + sample["ntokens_1"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size_regress,
        }

        return sample_size_regress , logging_output

    # @staticmethod
    # def reduce_metrics(logging_outputs) -> None:
    #     """Aggregate logging outputs from data parallel training."""
    #     id_tensor_list = [log.get("id", 0) for log in logging_outputs]
    #     target_tensor_list = [log.get("target", 0) for log in logging_outputs]
    #     cls_0_tensor_list = [log.get("cls_0", 0) for log in logging_outputs]
    #     cls_1_tensor_list = [log.get("cls_1", 0) for log in logging_outputs]
    #     dataset_list = [log.get("dataset", 0) for log in logging_outputs]
    #     # sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

    #     id_list, cls_0_list, cls_1_list, target_list = [], [], [], []


    #     for i in id_tensor_list:
    #         if i.dim() == 1:
    #             for j in i:
    #                 id_list.append(j.detach().cpu().numpy())
    #         else: # i.dim() == 0
    #             id_list.append(i.detach().cpu().numpy())

    #     for i in cls_0_tensor_list:
    #         if i.dim() == 2:
    #             for j in i:
    #                 cls_0_list.append(j.detach().cpu().numpy())
    #         else: # i.dim() == 1
    #             cls_0_list.append(i.detach().cpu().numpy())

    #     for i in cls_1_tensor_list:
    #         if i.dim() == 2:
    #             for j in i:
    #                 cls_1_list.append(j.detach().cpu().numpy())
    #         else: # i.dim() == 1
    #             cls_1_list.append(i.detach().cpu().numpy())

    #     for i in target_tensor_list:
    #         if i.dim() == 1:
    #             for j in i:
    #                 target_list.append(j.detach().cpu().numpy())
    #         else: # i.dim() == 0
    #             target_list.append(i.detach().cpu().numpy())

    #     df = pd.DataFrame({'cls_0': cls_0_list, 'cls_1': cls_1_list, 'target': target_list}, index=np.array(id_list))
    #     df.sort_index(inplace=True)

    #     cls_0_np = np.array(df['cls_0'].to_list()).astype('float32')
    #     cls_1_np = np.array(df['cls_1'].to_list()).astype('float32')
    #     unique_cls_0_np = np.array(list(set([tuple(t) for t in df['cls_0'].to_list()]))).astype('float32')
    #     unique_cls_1_np = np.array(list(set([tuple(t) for t in df['cls_1'].to_list()]))).astype('float32')

    #     # if not os.path.exists(self.datastore_path):
    #     #     os.makedirs(self.datastore_path)

    #     # np.save(f'{self.datastore_path}/cls_0', cls_0_np)
    #     # np.save(f'{self.datastore_path}/cls_1', cls_1_np)
    #     # np.save(f'{self.datastore_path}/cls_0_unique_mol', unique_cls_0_np)
    #     # np.save(f'{self.datastore_path}/cls_1_unique_pro', unique_cls_1_np)

    #     tmp_datastore_path = '/protein/users/v-qizhipei/datastore_tmp'

    #     if not os.path.exists(tmp_datastore_path):
    #         os.makedirs(tmp_datastore_path)
        
    #     np.save(f'{tmp_datastore_path}/cls_0', cls_0_np)
    #     np.save(f'{tmp_datastore_path}/cls_1', cls_1_np)
    #     np.save(f'{tmp_datastore_path}/cls_0_unique_mol', unique_cls_0_np)
    #     np.save(f'{tmp_datastore_path}/cls_1_unique_pro', unique_cls_1_np)

    #     metrics.log_scalar(
    #         "training_set_size", len(cls_0_np), round=5
    #     )

    #     metrics.log_scalar(
    #         "unique_mol_num", len(unique_cls_0_np), round=5
    #     )

    #     metrics.log_scalar(
    #         "unique_pro_num", len(unique_cls_1_np), round=5
    #     )

    @staticmethod
    def reduce_metrics(logging_outputs, append_args=None) -> None:
        """Aggregate logging outputs from data parallel training."""

        # sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
