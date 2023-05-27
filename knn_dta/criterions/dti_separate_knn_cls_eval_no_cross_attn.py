# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index

@register_criterion("dti_separate_knn_cls_eval_no_cross_attn")
class DTIRegressKNNCLSEvalNoCrossAttnLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) and regression training.
    """

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target


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
            "ntokens": sample["ntokens_0"] + sample["ntokens_1"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size_regress,
        }

        return sample_size_regress , logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, append_args=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        final_prediction_tensor_list = [log.get("final_prediction", 0) for log in logging_outputs]
        target_tensor_list = [log.get("target", 0) for log in logging_outputs]
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        final_prediction_list, target_list = [], []
        for i in final_prediction_tensor_list:
            if i.dim() == 1:
                for j in i:
                    final_prediction_list.append(j.detach().cpu().data)
            else: # i.dim() == 0
                final_prediction_list.append(i.detach().cpu().data)
        for i in target_tensor_list:
            if i.dim() == 1:
                for j in i:
                    target_list.append(j.detach().cpu().data)
            else: # i.dim() == 0
                target_list.append(i.detach().cpu().data)
        # round控制四舍五入的位数
        metrics.log_scalar(
            "MSE", mean_squared_error(final_prediction_list, target_list), round=5
        )
        metrics.log_scalar(
            "RMSE", math.sqrt(mean_squared_error(final_prediction_list, target_list)), round=5
        )

        metrics.log_scalar(
            "Pearson", pearsonr(final_prediction_list, target_list)[0], round=5
        )

        # print('C-index:', concordance_index(gold, pred))
        metrics.log_scalar(
            "C-index", concordance_index(target_list, final_prediction_list), round=5
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
