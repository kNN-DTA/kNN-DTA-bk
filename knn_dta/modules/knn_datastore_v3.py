import torch
import faiss
import numpy as np
# from torch_scatter import scatter
import time
import math
import faiss.contrib.torch_utils
import os


class KNN_Dstore_V3(object):

    def __init__(self, args):

        # self.half = args.fp16
        # self.dimension = args.decoder_embed_dim
        # self.dstore_size = args.dstore_size
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        # self.dstore_fp16 = args.dstore_fp16
        # self.temperature = args.knn_temperature
        # self.use_gpu_to_search = args.use_gpu_to_search
        # self.vocab_size = trg_vocab_size
        # self.only_use_max_idx = args.only_use_max_idx

        self.index, self.vals, self.index_mol, self.vals_mol, self.index_pro, self.vals_pro = self.setup_faiss(args)
        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.

        # set lambda
        self.set_lambda(args)

        # set temperature
        self.temperature_type = args.knn_temperature_type
        if self.temperature_type == 'fix':
            self.temperature = args.knn_temperature_value
            self.temperature_mol = args.knn_temperature_value
            self.temperature_pro = args.knn_temperature_value
        elif self.temperature_type == 'trainable':
            self.temperature = None
            self.temperature_mol = None
            self.temperature_pro = None
        else:
            self.temperature = None
            self.temperature_mol = None
            self.temperature_pro = None

        self.k_type = args.knn_k_type
  
        self.k = args.k
        self.k_mol = args.k_mol
        self.k_pro = args.k_pro


    def set_lambda(self, args):

        if not hasattr(args, 'knn_lambda_type'):
            return

        self.lambda_type = args.knn_lambda_type

        if self.lambda_type == 'fix':
            self.lambda_value = args.knn_lambda_value

        if self.lambda_type == 'trainable':
            self.lambda_value = None  # not generate lambda value in this class

    def get_lambda(self, step=None, distance=None):

        if self.lambda_type == 'fix':

            return self.lambda_value

        elif self.lambda_type == 'trainable':

            return None

    def get_temperature(self):
        if self.temperature_type == 'fix':
            return self.temperature
        else:
            return None

    def get_temperature_mol(self):
        if self.temperature_type == 'fix':
            return self.temperature_mol
        else:
            return None

    def get_temperature_pro(self):
        if self.temperature_type == 'fix':
            return self.temperature_pro
        else:
            return None

    def setup_faiss(self, args):
        if not args.datastore_path:
            raise ValueError('Cannot build a datastore without the data.')

        res = faiss.StandardGpuResources()
        faiss_cfg = faiss.GpuIndexFlatConfig()
        faiss_cfg.useFloat16 = False
        faiss_cfg.device = torch.cuda.current_device()

        cls_mol = np.load(os.path.join(args.datastore_path, 'cls_0.npy'))
        cls_pro = np.load(os.path.join(args.datastore_path, 'cls_1.npy'))
        cls_mol_pro = torch.from_numpy(np.c_[cls_mol, cls_pro]).cuda()
        
        label_list = [float(line.strip()) for line in open(f'{args.data}/label/train.label')]
        cls_label = torch.FloatTensor(label_list).unsqueeze(-1).cuda()
        # cls_label = torch.from_numpy(np.load(os.path.join(args.datastore_path, 'label.npy'))).cuda()

        gpu_index_flat = faiss.GpuIndexFlatL2(res, self.embed_dim * 2, faiss_cfg)
        gpu_index_flat.add(cls_mol_pro)

        cls_mol_unique = torch.from_numpy(np.load(os.path.join(args.datastore_path, 'cls_0_unique_mol.npy'))).cuda()
        cls_pro_unique = torch.from_numpy(np.load(os.path.join(args.datastore_path, 'cls_1_unique_pro.npy'))).cuda()

        gpu_index_flat_mol = faiss.GpuIndexFlatL2(res, self.embed_dim, faiss_cfg)
        gpu_index_flat_mol.add(cls_mol_unique)

        gpu_index_flat_pro = faiss.GpuIndexFlatL2(res, self.embed_dim, faiss_cfg)
        gpu_index_flat_pro.add(cls_pro_unique)

        return gpu_index_flat, cls_label, gpu_index_flat_mol, cls_mol_unique, gpu_index_flat_pro, cls_pro_unique

    def get_knns_label(self, queries):

        # move query to numpy, if faiss version < 1.6.5
        # numpy_queries = queries.detach().cpu().float().numpy()
        
        if self.args.train_subset == 'train':
            dists, knns = self.index.search(queries, self.k + 1)
            dists = dists[:, 1:]
            knns = knns[:, 1:]
        else:
            dists, knns = self.index.search(queries, self.k)
        return dists, knns

    def get_knns_mol(self, queries):

        # move query to numpy, if faiss version < 1.6.5
        # numpy_queries = queries.detach().cpu().float().numpy()
        
        if self.args.train_subset == 'train':
            dists, knns = self.index_mol.search(queries, self.k_mol + 1)
            dists = dists[:, 1:]
            knns = knns[:, 1:]
        else:
            dists, knns = self.index_mol.search(queries, self.k_mol)
        return dists, knns

    def get_knns_pro(self, queries):

        # move query to numpy, if faiss version < 1.6.5
        # numpy_queries = queries.detach().cpu().float().numpy()
        
        if self.args.train_subset == 'train':
            dists, knns = self.index_pro.search(queries, self.k_pro + 1)
            dists = dists[:, 1:]
            knns = knns[:, 1:]
        else:
            dists, knns = self.index_pro.search(queries, self.k_pro)
        return dists, knns


    def retrieve_label(self, queries):

        # queries are [Batch, Hid Size]
        # retrieve
        bsz = queries.size(0)
        dists, knns = self.get_knns_label(queries)
        value = torch.index_select(self.vals, 0, torch.flatten(knns)).reshape(bsz, self.k)

        return {'distance': dists, 'knn_index': knns, 'value': value}

    def retrieve_mol(self, queries):

        # queries are [Batch, Hid Size]
        # retrieve
        bsz = queries.size(0)
        dists, knns = self.get_knns_mol(queries)
        value = torch.index_select(self.vals_mol, 0, torch.flatten(knns)).reshape(bsz, self.k_mol, self.embed_dim)

        return {'distance': dists, 'knn_index': knns, 'value': value}

    def retrieve_pro(self, queries):

        # queries are [Batch, Hid Size]
        # retrieve
        bsz = queries.size(0)
        dists, knns = self.get_knns_pro(queries)
        value = torch.index_select(self.vals_pro, 0, torch.flatten(knns)).reshape(bsz, self.k_pro, self.embed_dim)

        return {'distance': dists, 'knn_index': knns, 'value': value}

