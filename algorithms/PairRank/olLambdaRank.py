# -*- coding: utf-8 -*-
# import itertools
import numpy as np
import sys
# import networkx as nx
import random
import math
import utils.rankings as rnk
import utils.evaluate as eval
from utils.net_utils import *
# from models.mlpmodel import MLPModel
# from numpy.linalg import multi_dot
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.PairRank.olRankNet import olRankNet
# import sys
import torch
# from backpack import backpack, extend
# from backpack.extensions import BatchGrad
from itertools import islice
# import torch.nn.functional as F

# TODO Change the InvA into the diagonal of the matrix, otherwise it would take too much time and space
# Pairwise Logistic Regression


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def logist(s):
    return 1 / (1 + np.exp(-s))


def logistic_func(theta, x):
    return float(1) / (1 + math.e ** (-x.dot(theta)))


def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))


def get_dcg(ordered_labels):
    return np.sum(
        (2 ** ordered_labels - 1) / np.log2(np.arange(ordered_labels.shape[0]) + 2)
    )


def get_idcg(complete_labels, max_len):
    return get_dcg(np.sort(complete_labels)[: -1 - max_len : -1])

# class pairwise_loss(torch.nn.Module):
#     def __init__(self, device):
#         super(pairwise_loss, self).__init__()
#         self.device = device

#     def forward(self, pred, weight=None):
#         n_pairs = int(len(pred) / 2)
#         pred_diff = pred[:n_pairs] - pred[n_pairs:]
#         target = torch.ones(n_pairs).to(self.device)
#         loss = F.binary_cross_entropy_with_logits(pred_diff, target, weight=weight)
#         return loss.mean()


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self):
#         self.doc_pair_history = []
#         self.doc_range = [0]
#         self.qid = []

#     def push(self, qid, doc_pairs):
#         self.qid.append(qid)

#         self.doc_pair_history.extend(list(doc_pairs))

#         # for i, pair in enumerate(doc_pairs):
#         #     self.doc_pair_history.append(pair)
#         self.doc_range.append(len(self.doc_pair_history))

#     def get_data(self, index):
#         s_i = self.doc_range[index]
#         e_i = self.doc_range[index + 1]
#         return self.doc_pair_history[s_i:e_i]

#     def __len__(self):
#         return len(self.doc_pair_history)

#     def __getitem__(self, index):
#         # return the observed pairs of a specific round: index
#         return self.doc_pair_history[index]


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i * n : (i + 1) * n] for i in range((len(list_in) + n - 1) // n)]


class olLambdaRank(olRankNet):
    def __init__(
        self,
        alpha,
        _lambda,
        refine,
        rank,
        update,
        learning_rate,
        learning_rate_decay,
        ind,
        mlp_dims,
        batch_size,
        epoch,
        *args,
        **kargs
    ):
        super().__init__(alpha,
        _lambda,
        refine,
        rank,
        update,
        learning_rate,
        learning_rate_decay,
        ind,
        mlp_dims,
        batch_size,
        epoch, *args, **kargs)

       

        self.history = {}
        self.init_history()
        self.decay_diff = []
        self.init_decay_diff()

    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
          'learning_rate': 0.1,
          'learning_rate_decay': 1.0
          })
        return parent_parameters

    def init_decay_diff(self):
        x, y = np.meshgrid(np.arange(self.n_results), np.arange(self.n_results))
        self.decay_diff = np.abs(1.0 / np.log2(x + 2) - 1.0 / np.log2(y + 2))

    def init_history(self):
        self.history["qids"] = []
        self.history["docs"] = []
        self.history["labels"] = []
        self.history["ranges"] = [0]
        self.history["idcgs"] = []
        self.history['doc_pairs'] = []

    def update_to_interaction(self, clicks):
        if np.any(clicks):
            self._update_to_clicks(clicks)

    def update_data(self, clicks):
        pairs = self.generate_pairs(clicks)

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

            # update dataset
            n_docs = self.ranking.shape[0]
            cur_k = np.minimum(n_docs, self.n_results)
            included = np.ones(cur_k, dtype=np.int32)
            if not clicks[-1]:
                included[1:] = np.cumsum(clicks[::-1])[:0:-1]
            last_obs = sum((included > 0) * 1)
            query_index = self.get_query_global_index(self._last_query_id, self._train_query_ranges)
            self.history["qids"].append(self._last_query_id)
            self.history['doc_pairs'].extend(list(pairs + len(self.history["docs"])))
            self.history["docs"].extend(self.ranking[:last_obs] + query_index)
            self.history["labels"].extend(clicks[:last_obs])
            self.history["ranges"].append(self.history["ranges"][-1] + last_obs)
            self.history["idcgs"].extend(
                [eval.get_idcg(clicks[:last_obs], last_obs)] * last_obs
            )

            assert len(self.history["docs"]) == len(self.history["idcgs"])
            assert len(self.history["qids"]) == len(self.history["ranges"]) - 1.0
            torch.cuda.empty_cache()
            self.update_to_history()

    def _update_to_clicks(self, clicks):
        self.update_data(clicks)
        self.update_to_history()
        
    def update_to_history(self):

        # initialize the setting
        num_data = len(self.history['doc_pairs'])
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self._lambda / num_data,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.learning_rate_decay
        )
        self.model.zero_grad()
        train_feature = self._train_features[self.history["docs"]]
        train_feature_torch = torch.tensor(train_feature).to(self.device)

        # reset model
        self.model.reset_parameters()

        for i in range(self.epoch):

            # total_loss = 0
            j = 0
            self.batch = partition(
                list(np.arange(len(self.history['doc_pairs']))), self.batch_size
            )
            if len(self.batch[-1]) < 0.5 * self.batch_size and len(self.batch) > 1:
                self.batch = self.batch[:-1]

            if self.ind:
                num_batch = len(self.batch)
            else:
                num_batch = min(10, len(self.batch))
            # for batch in data_generator:
            for j in range(num_batch):
                batch_index = np.array(self.batch[j])
                doc_pair_index = np.array(self.history['doc_pairs'])[batch_index]
                # query_feature = self._train_features[doc_pair_index]

                # get the predicted scores for all the observed historical documents
                scores = -self.model.predict(train_feature_torch.float()).reshape(-1)
                # get the predicted ranking of each doc
                rankings = rnk.rank_queries(
                    scores, np.array(self.history["ranges"]), inverted=True
                )
                self.model.zero_grad()

                # pair_ranks = rankings[batch]
                pair_ranks = rankings[doc_pair_index]
                decay_diff = self.decay_diff[pair_ranks[:, 0], pair_ranks[:, 1]].reshape(-1)
                delta_ndcg = torch.tensor(decay_diff* 1.0 / np.array(self.history["idcgs"])[doc_pair_index[:, 0]]
                ).to(self.device)

                query_feature = self._train_features[
                    np.array(self.history["docs"])[doc_pair_index]
                ]
                query_feature_torch = torch.tensor(query_feature).to(self.device)
                pred = self.model(query_feature_torch.float()).view(-1, 2)
                target = torch.ones(len(pred)).to(self.device)
                loss = self.bpr_loss(pred, target, weight=delta_ndcg)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()
                j += 1
            scheduler.step()

