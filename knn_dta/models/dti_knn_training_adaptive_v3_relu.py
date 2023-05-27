import os
from typing import Any, Dict

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor

import numpy as np
import math
###################################
import faiss
import faiss.contrib.torch_utils
###################################
from fairseq import utils
from fairseq.modules import MultiheadAttention
from fairseq import checkpoint_utils
from fairseq.models.custom_roberta.model import (
    RobertaModel,
    RobertaEncoder,
    DTIRobertaEncoder,
    RobertaClassificationHead,
    base_architecture as roberta_base_architecture
)

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules import GradMultiply
from fairseq.modules.knn_datastore_v3 import KNN_Dstore_V3

DEFAULT_MAX_MOLECULE_POSITIONS = 512
DEFAULT_MAX_PROTEIN_POSITIONS = 1024

logger = logging.getLogger(__name__)

@register_model("dti_knn_training_adaptive_v3_relu")
# class RobertaDTI(RobertaModel):
class RobertaDTIKNNTrainingAdaptiveVersion3Relu(BaseFairseqModel):
    def __init__(self, args, encoder_0, encoder_1, classification_head, knn_meta_network):
        # TODO
        super().__init__()
        self.args = args
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        # We follow BERT's random weight initialization
        
        # TODO add init_bert_params for newly added modules

        # self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()
        self.classification_heads[getattr(args, "classification_head_name", "sentence_classification_head")] = classification_head
        
        # self.cls_tmp_0 = cls_tmp_0
        # self.cls_tmp_1 = cls_tmp_1
        # self.gpu_index_flat_0 = gpu_index_flat_0
        # self.gpu_index_flat_1 = gpu_index_flat_1
        # self.gpu_index_flat = gpu_index_flat
        # self.cls_label = cls_label

        # self.knn_meta_network = KnnMetaNetwork(args, self.gpu_index_flat, self.cls_label)
        self.knn_meta_network = knn_meta_network
        if args.apply_layer_norm:
            self.layer_norm_mol = nn.LayerNorm(args.encoder_embed_dim)
            self.layer_norm_pro = nn.LayerNorm(args.encoder_embed_dim)
       
        
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--grad-multiply", type=float, metavar="D", default=1, help="Apply different lr on backbone and classification head"
        )
        parser.add_argument(
            "--knn-only-mlp",
            action="store_true",
            default=False,
            help="Only use mlp layer in knn embedding transformer",
        )
        parser.add_argument(
            "--pretrained-molecule-protein-roberta-checkpoint",
            type=str,
            metavar="STR",
            help="roberta model to use for initializing molecule encoder",
        )
        parser.add_argument(
            "--init-molecule-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into protein encoder",
        )
        parser.add_argument(
            "--init-protein-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into molecule encoder",
        )
        parser.add_argument("--k", type=int, metavar="16", default=16, help="k nearest neightbor for search")
        parser.add_argument("--k-mol", type=int, metavar="16", default=16, help="k nearest neightbor for search")
        parser.add_argument("--k-pro", type=int, metavar="16", default=16, help="k nearest neightbor for search")
        parser.add_argument("--model-eval", action="store_true", default=False, help="if set, open the model eval mode for pre-trained fixed model")
        parser.add_argument("--apply-layer-norm", action="store_true", default=False, help="if set, apply layer norm for aggregated tensor")
        parser.add_argument("--faiss-metric-type", default=None, type=str)
        parser.add_argument("--knn-sim-func", default=None, type=str)

        parser.add_argument("--knn-lambda-type", default="fix", type=str)
        parser.add_argument("--knn-lambda-value", default=0.5, type=float)
        parser.add_argument("--knn-lambda-net-hid-size", default=0, type=int)

        parser.add_argument("--label-value-as-feature", default=False, action="store_true")
        # parser.add_argument("--relative-label-count", default=False, action="store_true")
        parser.add_argument("--knn-net-dropout-rate", default=0.5, type=float)

        parser.add_argument("--knn-temperature-type", default="fix", type=str)
        parser.add_argument("--knn-temperature-value", default=10, type=float)
        # parser.add_argument("--knn-temperature-value-mol", default=10, type=float)
        # parser.add_argument("--knn-temperature-value-pro", default=10, type=float)
        parser.add_argument("--knn-temperature-net-hid-size", default=0, type=int)
        # we add 4 arguments for trainable k network
        parser.add_argument("--knn-k-type", default="fix", type=str)
        # parser.add_argument("--max-k", default=32, type=int)
        # parser.add_argument("--max-k-mol", default=32, type=int)
        # parser.add_argument("--max-k-pro", default=32, type=int)
        parser.add_argument("--knn-k-net-hid-size", default=0, type=int)
        parser.add_argument("--knn-k-net-dropout-rate", default=0, type=float)

        # we add 3 arguments for trainable k_with_lambda network
        parser.add_argument("--k-lambda-net-hid-size-mol", type=int, default=0)
        parser.add_argument("--k-lambda-net-hid-size-pro", type=int, default=0)
        parser.add_argument("--k-lambda-net-hid-size", type=int, default=0)
        parser.add_argument("--k-lambda-net-dropout-rate", type=float, default=0.0)
        parser.add_argument("--gumbel-softmax-temperature", type=float, default=1)

        parser.add_argument(
            "--fix-classification-head",
            action="store_true",
            default=False,
            help="Fix classification head during training",
        )
        parser.add_argument(
            "--cls-residual",
            action="store_true",
            default=False,
            help="Add residual connection to cls",
        )

        parser.add_argument(
            "--datastore-path",
            type=str,
            metavar="STR",
            default="tmp",
            help="Datastore path",
        )

        parser.add_argument(
            "--max-positions-molecule", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--max-positions-protein", type=int, help="number of positional embeddings to learn"
        )

        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-knn-layers", type=int, metavar="L", help="num knn wo embedding transformer encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )

        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        assert hasattr(args, "pretrained_molecule_protein_roberta_checkpoint"), (
            "You must specify a path for --pretrained-molecule-protein-roberta-checkpoint to use "
            "--arch dti_knn_training_adaptive"
        )

        assert not (
            getattr(args, "init_molecule_encoder_only", False)
            and getattr(args, "init_protein_encoder_only", False)
        ), "Only one of --init-molecule-encoder-only and --init-protein-encoder-only can be set."
        # make sure all arguments are present. Missing arguments will be filled by this function.
        base_architecture(args)

        if not hasattr(args, 'max_positions_molecule'):
            args.max_positions_molecule = DEFAULT_MAX_MOLECULE_POSITIONS
        if not hasattr(args, 'max_positions_protein'):
            args.max_positions_protein = DEFAULT_MAX_PROTEIN_POSITIONS

        encoder_0 = cls.build_molecule_encoder(args, task.source_dictionary_0)
        encoder_1 = cls.build_protein_encoder(args, task.source_dictionary_1)


        for param in encoder_0.parameters():
            param.requires_grad = False
        
        for param in encoder_1.parameters():
            param.requires_grad = False

        classification_head = cls.build_classification_head(args)

        if getattr(args, "fix_classification_head", False):
            for param in classification_head.parameters():
                param.requires_grad = False

        knn_meta_network = cls.build_knn_meta_network(args)

        return cls(args, encoder_0, encoder_1, classification_head, knn_meta_network)

    @classmethod
    def build_molecule_encoder(cls, args, src_dict):
        return MoleculeEncoderFromPretrainedRoberta(args, src_dict)

    @classmethod
    def build_protein_encoder(cls, args, src_dict):
        return ProteinEncoderFromPretrainedRoberta(args, src_dict)
    
    @classmethod
    def build_classification_head(cls, args):
        return ClassificationHeadFromPretrained(args)

    @classmethod
    def build_knn_meta_network(cls, args):
        return KnnMetaNetwork(args)


    def forward(
        self,
        src_tokens_0,
        src_tokens_1,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs
    ):
        # torch.autograd.set_detect_anomaly(True)
        if classification_head_name is not None:
            features_only = True
        if self.args.model_eval:
            self.encoder_0.eval()
            self.encoder_1.eval()

        x_0, extra_0 = self.encoder_0(src_tokens_0, features_only, return_all_hiddens, **kwargs)
        x_1, extra_1 = self.encoder_1(src_tokens_1, features_only, return_all_hiddens, **kwargs)

        x_0_query = x_0[:, 0, :].clone().detach()
        x_1_query = x_1[:, 0, :].clone().detach()

        if classification_head_name is not None:
            cls_0_agg_neighbor = self.knn_meta_network(model_prediction=None, query=x_0_query, mode='mol')
            cls_1_agg_neighbor = self.knn_meta_network(model_prediction=None, query=x_1_query, mode='pro')
            cls_0_agg_neighbor = self.layer_norm_mol(cls_0_agg_neighbor)
            cls_1_agg_neighbor = self.layer_norm_pro(cls_1_agg_neighbor)
            # x = torch.cat((x_0[:, 0, :], x_1[:, 0, :]), 1).unsqueeze(1)
            x = torch.cat((cls_0_agg_neighbor, cls_1_agg_neighbor), 1).unsqueeze(1)
            if isinstance(x, Tensor):
                x = GradMultiply.apply(x, self.args.grad_multiply)
            x = self.classification_heads[classification_head_name](x)
        
            x = self.knn_meta_network(model_prediction=x, query=torch.cat((x_0_query, x_1_query), dim=1), mode='label')

        return x, extra_0, extra_1



def upgrade_state_dict_with_pretrained_model_weights(
    state_dict: Dict[str, Any], pretrained_model_checkpoint: str, part: str
) -> Dict[str, Any]:
    """
    Load Roberta weights into a Roberta encoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_model_checkpoint: checkpoint to load roberta weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current roberta encoder
            and the pretrained_model_checkpoint
    """
    if not os.path.exists(pretrained_model_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_model_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_model_checkpoint)
    pretrained_state_dict = state["model"]
    # if load two encoder's parameter separately
    # if 'encoder.sentence_encoder.layernorm_embedding.weight' in pretrained_state_dict.keys():
    #     pretrained_state_dict['encoder.sentence_encoder.emb_layer_norm.weight'] = pretrained_state_dict.pop('encoder.sentence_encoder.layernorm_embedding.weight')
    # if 'encoder.sentence_encoder.layernorm_embedding.bias' in pretrained_state_dict.keys():
    #     pretrained_state_dict['encoder.sentence_encoder.emb_layer_norm.bias'] = pretrained_state_dict.pop('encoder.sentence_encoder.layernorm_embedding.bias')
    i = 0
    for key in pretrained_state_dict.keys():
        if key.startswith(part):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            subkey = key[len(part) + 1:]
            if subkey in state_dict.keys():
                state_dict[subkey] = pretrained_state_dict[key]
                i += 1

    logger.info(f'load {i} values from the {part} part of pretrained molecule protein model')
    return state_dict


class MoleculeEncoderFromPretrainedRoberta(DTIRobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary, args.max_positions_molecule)

        roberta_loaded_state_dict = upgrade_state_dict_with_pretrained_model_weights(
            state_dict=self.state_dict(),
            pretrained_model_checkpoint=args.pretrained_molecule_protein_roberta_checkpoint,
            part="encoder_0",
        )
        self.load_state_dict(roberta_loaded_state_dict, strict=True)


class ProteinEncoderFromPretrainedRoberta(DTIRobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary, args.max_positions_protein)

        roberta_loaded_state_dict = upgrade_state_dict_with_pretrained_model_weights(
            state_dict=self.state_dict(),
            pretrained_model_checkpoint=args.pretrained_molecule_protein_roberta_checkpoint,
            part="encoder_1",
        )
        self.load_state_dict(roberta_loaded_state_dict, strict=True)

class ClassificationHeadFromPretrained(RobertaClassificationHead):
    def __init__(self, args, inner_dim=None):
        super().__init__(
            input_dim=2 * args.encoder_embed_dim, # concat interaction tokens
            inner_dim=inner_dim or args.encoder_embed_dim,
            num_classes=args.num_classes,
            activation_fn=args.pooler_activation_fn,
            pooler_dropout=args.pooler_dropout,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            do_spectral_norm=args.spectral_norm_classification_head,
        )
        classification_head_loaded_state_dict = upgrade_state_dict_with_pretrained_model_weights(
            state_dict=self.state_dict(),
            pretrained_model_checkpoint=args.pretrained_molecule_protein_roberta_checkpoint,
            part="classification_heads.sentence_classification_head",
        )
        self.load_state_dict(classification_head_loaded_state_dict, strict=True)

class KnnMetaNetwork(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.knn_datastore = KNN_Dstore_V3(args)
        # self.use_knn_datastore = args.use_knn_datastore
        self.knn_lambda_type = args.knn_lambda_type
        self.knn_temperature_type = args.knn_temperature_type
        self.knn_k_type = args.knn_k_type
        self.label_value_as_feature = args.label_value_as_feature

        if self.knn_lambda_type == "trainable" and self.knn_k_type == "trainable":

            # TODO another network to predict k and lambda at the same time without gumbel softmax
            self.retrieve_result_to_k_and_lambda = nn.Sequential(
                nn.Linear(args.k if not self.label_value_as_feature else args.k * 2,
                          args.k_lambda_net_hid_size),
                nn.ReLU(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(args.k_lambda_net_hid_size, 1 + args.k),
                nn.Softmax(dim=-1),  # [0 neighbor, 1 neighbor, 2 neighbor prob, 3, ... , k]
            )

            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, : args.k], gain=0.01)

            if self.label_value_as_feature:
                # TODO gain=0.1 need to be tuned
                nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, args.k:], gain=0.1)

            self.retrieve_result_to_k_and_lambda_mol = nn.Sequential(
                nn.Linear(args.k_mol, args.k_lambda_net_hid_size_mol),
                nn.ReLU(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(args.k_lambda_net_hid_size_mol, 1 + args.k_mol),
                nn.Softmax(dim=-1),  # [0 neighbor, 1 neighbor, 2 neighbor prob, 3, ... , k]
            )

            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda_mol[0].weight[:, : args.k_mol], gain=0.01)

            self.retrieve_result_to_k_and_lambda_pro = nn.Sequential(
                nn.Linear(args.k_pro, args.k_lambda_net_hid_size_pro),
                nn.ReLU(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(args.k_lambda_net_hid_size_pro, 1 + args.k_pro),
                nn.Softmax(dim=-1),  # [0 neighbor, 1 neighbor, 2 neighbor prob, 3, ... , k]
            )

            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda_pro[0].weight[:, : args.k_pro], gain=0.01)
            
    
    def forward(self, model_prediction, query, mode='label'):
        if mode == 'label':
            knn_search_result = self.knn_datastore.retrieve_label(query)

            D = knn_search_result['distance']
            I = knn_search_result['knn_index']
            V = knn_search_result['value']
            
            if self.label_value_as_feature:
                network_inputs = torch.cat((D.detach(), V.detach()), dim=-1)
            else:
                network_inputs = D.detach()
            
            if self.knn_lambda_type == "trainable" and self.knn_k_type == 'trainable':
                network_outputs = self.retrieve_result_to_k_and_lambda(network_inputs)
                knn_V = torch.cat((model_prediction, V), dim=-1)
                final_result = torch.sum(network_outputs * knn_V, dim=-1)

        elif mode == 'mol':
            knn_search_result = self.knn_datastore.retrieve_mol(query)
            D = knn_search_result['distance']
            I = knn_search_result['knn_index']
            V = knn_search_result['value']
            network_inputs = D.detach()
            if self.knn_lambda_type == "trainable" and self.knn_k_type == 'trainable':
                network_outputs = self.retrieve_result_to_k_and_lambda_mol(network_inputs)
                knn_V = torch.cat((query.unsqueeze(-2), V), dim=-2)
                final_result = torch.sum(network_outputs.unsqueeze(-1) * knn_V, dim=-2)
            

        elif mode == 'pro':
            knn_search_result = self.knn_datastore.retrieve_pro(query)
            D = knn_search_result['distance']
            I = knn_search_result['knn_index']
            V = knn_search_result['value']
            network_inputs = D.detach()
            if self.knn_lambda_type == "trainable" and self.knn_k_type == 'trainable':
                network_outputs = self.retrieve_result_to_k_and_lambda_pro(network_inputs)
                knn_V = torch.cat((query.unsqueeze(-2), V), dim=-2)
                final_result = torch.sum(network_outputs.unsqueeze(-1) * knn_V, dim=-2)
        
        return final_result

@register_model_architecture(
    "dti_knn_training_adaptive_v3_relu", "dti_knn_training_adaptive_v3_relu"
)
def base_architecture(args):
    roberta_base_architecture(args)




