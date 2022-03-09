# -*- coding: utf-8 -*-
# from scipy.optimize import minimize
import itertools
import numpy as np
# import networkx as nx
import random

# import math
import utils.rankings as rnk
from utils.net_utils import *
from models.mlpmodel import MLPModel
from algorithms.basiconlineranker import BasicOnlineRanker
import sys

# import os
import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from algorithms.PairRank.PairRank import PairRank

# import gc
# from functools import reduce
# import operator as op
# from itertools import islice
import torch.nn.functional as F
from utils.pairrank_utils import chunk

# import time


class pairwise_loss(torch.nn.Module):
    def __init__(self, device):
        super(pairwise_loss, self).__init__()
        self.device = device

    def forward(self, pred, weight=None):
        n_pairs = int(len(pred) / 2)
        pred_diff = pred[:n_pairs] - pred[n_pairs:]
        target = torch.ones(n_pairs).to(self.device)
        loss = F.binary_cross_entropy_with_logits(pred_diff, target, weight=weight)
        return loss.mean()


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i * n: (i + 1) * n] for i in range((len(list_in) + n - 1) // n)]


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.doc_pair_history = []
        self.doc_range = [0]
        self.qid = []

    def push(self, qid, doc_pairs):
        self.qid.append(qid)

        self.doc_pair_history.extend(list(doc_pairs))

        # for i, pair in enumerate(doc_pairs):
        #     self.doc_pair_history.append(pair)
        self.doc_range.append(len(self.doc_pair_history))

    def get_data(self, index):
        s_i = self.doc_range[index]
        e_i = self.doc_range[index + 1]
        return self.doc_pair_history[s_i:e_i]

    def __len__(self):
        return len(self.doc_pair_history)

    def __getitem__(self, index):
        # return the observed pairs of a specific round: index
        return self.doc_pair_history[index]


class olRankNet(PairRank):
    def __init__(self, alpha, _lambda, refine, rank, update, learning_rate, learning_rate_decay, ind, mlp_dims,
                 batch_size, epoch, *args, **kargs):
        super().__init__(alpha, _lambda, refine, rank, update, learning_rate, learning_rate_decay, ind, *args, **kargs)

        self.alpha = alpha
        self._lambda = _lambda
        self.refine = refine
        self.rank = rank
        self.update = update
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.ind = ind
        self.mlp_dims = mlp_dims
        self.device = get_device()
        self.batch_size = batch_size
        self.epoch = epoch

        self.model = extend(
            MLPModel(n_features=self.n_features, mlp_dims=self.mlp_dims, lr=self.learning_rate, ).to(self.device))
        self.total_param = self.model.total_param
        self.grad = None
        self.g = None
        self.score = None
        self.batch = None

        # corvariance matrix updated and used in GPU
        if update == "gd":
            self.Ainv = ((1 / self._lambda) * torch.eye(self.total_param, dtype=torch.float)).to(self.device)
        elif update == "gd_diag":
            self.A = (self._lambda * torch.ones(self.total_param, dtype=torch.float)).to(self.device)
        else:
            print("Update method is not supported.")
            sys.exit()

        # self.data = Dataset()
        # self.history = {}
        # self.pair_index = []

    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({'learning_rate': 0.1, 'learning_rate_decay': 1.0})
        return parent_parameters

    def get_test_rankings(self, features, query_ranges, inverted=True):
        with torch.no_grad():
            scores = -self.model.predict(torch.tensor(features, dtype=torch.float).to(self.device)).reshape(-1)
        return rnk.rank_multiple_queries(scores, query_ranges, inverted=inverted, n_results=self.n_results)

    def get_lcb(self, query_feat):
        n_doc = len(query_feat)
        feature_torch = torch.tensor(query_feat, dtype=torch.float).to(self.device)
        self.score = self.model(feature_torch).view(-1)
        sum_score = torch.sum(self.score)
        with backpack(BatchGrad()):
            sum_score.backward()
        self.grad = torch.cat([p.grad_batch.view(n_doc, -1) for p in self.model.parameters()], dim=1)

        # get the pairwise score
        self.score = self.score.view(-1, 1)
        prob_est = 1.0 / (1.0 + torch.exp(-(self.score - self.score.T)))

        max_size = 50
        if n_doc < max_size:
            block = n_doc
        else:
            block = max_size

        index_list = chunk(range(n_doc), block)
        s_index = 0
        uncertainty = torch.zeros(n_doc ** 2, dtype=torch.float).to(self.device)
        for _, l in enumerate(index_list):
            index = np.array(l)
            pairwise_feat = (self.grad[index, np.newaxis] - self.grad).reshape(-1, self.total_param)
            e_index = s_index + len(pairwise_feat)
            
            if self.update == "gd":
                uncertainty[s_index:e_index] = torch.sqrt(torch.diag(torch.matmul(torch.matmul(pairwise_feat, self.Ainv), torch.transpose(pairwise_feat, 0, 1))))
            elif self.update == "gd_diag":
                uncertainty[s_index:e_index] = torch.sqrt(torch.sum((pairwise_feat ** 2) / self.A, dim=1))
            else:
                print("Unsupported update method")
                sys.exit()
                
            s_index = e_index

        prob_est -= self.alpha * (uncertainty.view(n_doc, n_doc))

        lcb_matrix = prob_est.data.cpu().numpy()
        del prob_est
        del uncertainty
        torch.cuda.empty_cache()

        return lcb_matrix

    def _create_train_ranking(self, query_id, query_feat, inverted):
        lcb_matrix = self.get_lcb(query_feat)
        partition, sorted_list, certain_edges = self.get_partitions(lcb_matrix)
        self._last_query_feat = query_feat
        self.ranking = self.get_ranking(lcb_matrix, sorted_list, partition)

        topK_index = self.ranking[: self.n_results]
        self.g = self.grad[topK_index]

        return self.ranking

    def update_to_interaction(self, clicks):
        if np.any(clicks):
            self._update_to_clicks(clicks)

    def _update_to_clicks(self, clicks):
        # generate all pairs from the clicks
        pairs = self.generate_pairs(clicks)
        # update the model if we have valid observed pairs
        if len(pairs) != 0:
            
            diff_g = self.g[pairs[:, 0]] - self.g[pairs[:, 1]]
            if self.update == "gd":
                for i in range(len(diff_g)):
                    g = diff_g[i]
                    self.Ainv -= (torch.matmul(torch.matmul(torch.matmul(self.Ainv, g.view(self.total_param, -1)), g.view(-1, self.total_param)), self.Ainv))/(1 + torch.matmul(torch.matmul(g.view(-1, self.total_param), self.Ainv), g.view(self.total_param, -1)))
            elif self.update == "gd_diag":
                self.A += torch.sum(diff_g * diff_g, dim=0)
            else:
                print("Unsupported update method")
                sys.exit()
            self.update_history_data(pairs)
            self.update_to_history()

    def bpr_loss(self, pred, target, weight=None):

        pred_diff = pred[:, 0] - pred[:, 1]
        loss = F.binary_cross_entropy_with_logits(pred_diff, target, weight=weight, reduction="none")
        return loss.mean()

    def update_to_history(self):
        
        # num_data = len(self.doc_pair_history)
        # self.n_data
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self._lambda / self.n_data, )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.learning_rate_decay)
        self.model.zero_grad()
        self.model.reset_parameters()
        for i in range(self.epoch):
            self.batch = partition(list(np.arange(self.n_data)), self.batch_size)
    
            if self.ind:
                num_batch = len(self.batch)
            else:
                num_batch = min(10, len(self.batch))
            for j in range(num_batch):
                batch_index = np.array(self.batch[j])
                doc_pair_index = np.array(self.doc_pair_history)[batch_index]
                query_feature = self._train_features[doc_pair_index]
                query_feature = torch.tensor(query_feature).to(self.device)
                pred = self.model(query_feature.float()).view(-1, 2)
                size = len(pred)
                target = torch.ones(size).to(self.device)
                loss = self.bpr_loss(pred, target)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()
            scheduler.step()  # t2 = time.time()  # print("Num of Batch: {} Total training time {}".format(len(self.batch), t2 - t1))