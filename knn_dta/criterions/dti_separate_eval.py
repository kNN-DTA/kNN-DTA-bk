# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from pandas import DataFrame

@register_criterion("dti_separate_eval")
class DTISeparateEvalCriterion(FairseqCriterion):
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
        # fmt: on

    def forward(self, model, sample, reduce=True):
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

        logits, _, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction="sum")

        # logging_output = {
        #     "loss": loss.data,
        #     "ntokens": sample["ntokens_0"] + sample["ntokens_1"],
        #     "nsentences": sample_size,
        #     "sample_size": sample_size,
        # }
        logging_output = {
            "id": sample["id"],
            "prediction": logits,
            "target": targets,
            "ntokens": sample["ntokens_0"] + sample["ntokens_1"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, append_args=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        prediction_tensor_list = [log.get("prediction", 0) for log in logging_outputs]
        target_tensor_list = [log.get("target", 0) for log in logging_outputs]
        id_tensor_list = [log.get("id", 0) for log in logging_outputs]
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        prediction_list, target_list, id_list = [], [], []

        # for i in prediction_tensor_list:
        #     if i.dim() == 1:
        #         for j in i:
        #             prediction_list.append(j.detach().cpu().data)
        #     else: # i.dim() == 0
        #         prediction_list.append(i.detach().cpu().data)
        # for i in target_tensor_list:
        #     if i.dim() == 1:
        #         for j in i:
        #             target_list.append(j.detach().cpu().data)
        #     else: # i.dim() == 0
        #         target_list.append(i.detach().cpu().data)

        for i in prediction_tensor_list:
            if i.dim() == 1:
                for j in i:
                    prediction_list.append(j.detach().cpu().numpy())
            else: # i.dim() == 0
                prediction_list.append(i.detach().cpu().numpy())
        for i in target_tensor_list:
            if i.dim() == 1:
                for j in i:
                    target_list.append(j.detach().cpu().numpy())
            else: # i.dim() == 0
                target_list.append(i.detach().cpu().numpy())
        for i in id_tensor_list:
            if i.dim() == 1:
                for j in i:
                    id_list.append(j.detach().cpu().numpy())
            else: # i.dim() == 0
                id_list.append(i.detach().cpu().numpy())

        # round控制四舍五入的位数
        metrics.log_scalar(
            "MSE", mean_squared_error(prediction_list, target_list), round=5
        )
        metrics.log_scalar(
            "RMSE", math.sqrt(mean_squared_error(prediction_list, target_list)), round=5
        )
        metrics.log_scalar(
            "Pearson", pearsonr(prediction_list, target_list)[0], round=5
        )
        metrics.log_scalar(
            "C-index", concordance_index(target_list, prediction_list), round=5
        )

        df = DataFrame({'id': id_list, "Pred": prediction_list, "Gold": target_list})
        df.sort_values(by=['id'], ascending=(True), inplace=True)
        df.to_csv(append_args.output_fn, sep='\t', index=False, columns=['Pred', 'Gold'])


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
