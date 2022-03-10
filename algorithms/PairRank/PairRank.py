# -*- coding: utf-8 -*-
from scipy.optimize import minimize
import itertools
import numpy as np
import networkx as nx
from numpy.linalg import multi_dot
import utils.rankings as rnk
from models.linearmodel import LinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
import sys
from utils.pairrank_utils import update_nodes, update_edges, logist, logistic_func, safe_ln


class PairRank(BasicOnlineRanker):
    def __init__(self, alpha, _lambda, refine, rank, update, learning_rate, learning_rate_decay, ind, *args, **kargs):
        super(PairRank, self).__init__(*args, **kargs)

        self.alpha = alpha
        self._lambda = _lambda
        self.refine = refine
        self.rank = rank
        self.update = update
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.ind = ind
        self.A = self._lambda * np.identity(self.n_features)
        self.InvA = np.linalg.pinv(self.A)
        self.model = LinearModel(n_features=self.n_features, learning_rate=learning_rate, learning_rate_decay=1,
                                 n_candidates=1, )
        self.history = {}
        self.n_data = 0
        self.doc_pair_history = None

    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({"learning_rate": 0.1, "learning_rate_decay": 1.0})
        return parent_parameters

    def get_test_rankings(self, features, query_ranges, inverted=True):
        scores = -self.model.score(features)
        return rnk.rank_multiple_queries(scores, query_ranges, inverted=inverted, n_results=self.n_results)

    def cost_func_reg(self, theta, x, y):
        log_func_v = logistic_func(theta, x)
        step1 = y * safe_ln(log_func_v)
        step2 = (1 - y) * safe_ln(1 - log_func_v)
        final = (-step1 - step2).mean()
        final += self._lambda * theta.dot(theta)
        return final

    def log_gradient_reg(self, theta, x, y):
        # n = len(y)
        first_calc = logistic_func(theta, x) - y
        final_calc = first_calc.T.dot(x) / len(y)
        reg = 2 * self._lambda * theta
        final_calc += reg
        return final_calc

    def get_lcb(self, query_feat):
        if self.update == "gd_diag":
            Id = np.identity(self.InvA.shape[0])
            InvA = np.multiply(self.InvA, Id)
        elif self.update == "gd":
            InvA = self.InvA
        else:
            print("Update method is not supported.")
            sys.exit()

        pairwise_feat = (query_feat[:, np.newaxis] - query_feat).reshape(-1, self.n_features)
        pairwise_estimation = self.model.score(pairwise_feat)
        n_doc = len(query_feat)
        prob_est = logist(pairwise_estimation).reshape(n_doc, n_doc)
        for i in range(n_doc):
            for j in range(i + 1, n_doc):
                feat = query_feat[i] - query_feat[j]
                uncertainty = self.alpha * np.sqrt(np.dot(np.dot(feat, InvA), feat.T))
                prob_est[i, j] -= uncertainty
                prob_est[j, i] -= uncertainty
        return prob_est

    def get_partitions(self, lcb_matrix):
        n_nodes = len(lcb_matrix)
        certain_edges = set()
        for i in range(n_nodes):
            indices = [k for k, v in enumerate(lcb_matrix[i]) if v > 0.5]
            for j in indices:
                certain_edges.add((i, j))

        # refine the certain edges: remove the cycles between partitions.
        if self.refine:
            nodes = np.array(range(n_nodes))
            certainG = nx.DiGraph()
            certainG.add_nodes_from(nodes)
            certainG.add_edges_from(certain_edges)

            for n in certainG.nodes():
                a = nx.algorithms.dag.ancestors(certainG, n)
                for k in a:
                    certain_edges.add((k, n))

        # cut the complete graph by the certain edges
        uncertainG = nx.complete_graph(n_nodes)
        uncertainG.remove_edges_from(certain_edges)
        # get all the connected component by the uncertain edges
        sn_list = list(nx.connected_components(uncertainG))
        n_sn = len(sn_list)
        super_nodes = {}
        for i in range(n_sn):
            super_nodes[i] = sn_list[i]
        # create inv_cp to store the cp_id for each node
        inv_sn = {}
        for i in range(n_sn):
            for j in super_nodes[i]:
                inv_sn[j] = i
        super_edges = {}
        for i in range(n_sn):
            super_edges[i] = set([])
        for i, e in enumerate(certain_edges):
            start_node, end_node = e[0], e[1]
            start_sn, end_sn = inv_sn[start_node], inv_sn[end_node]
            if start_sn != end_sn:
                super_edges[start_sn].add(end_sn)

        SG = nx.DiGraph(super_edges)
        flag = True
        cycle = []
        try:
            cycle = nx.find_cycle(SG)
        except Exception as e:
            # print(e)
            flag = False

        while flag:
            # get all candidate nodes
            candidate_nodes = set()
            for c in cycle:
                n1, n2 = c
                candidate_nodes.add(n1)
                candidate_nodes.add(n2)
            new_id = min(candidate_nodes)
            # update the edges
            super_edges = update_edges(super_edges, candidate_nodes, new_id)
            super_nodes = update_nodes(super_nodes, candidate_nodes, new_id)
            # print("=======After merge {}=======".format(cycle))
            # print("super_edges: ", super_edges)
            # print("super_nodes: ", super_nodes)

            SG = nx.DiGraph(super_edges)
            try:
                cycle = nx.find_cycle(SG)
            except Exception as e:
                # print(e)
                flag = False

        sorted_list = list(nx.topological_sort(SG))

        return super_nodes, sorted_list, certain_edges

    def get_ranking(self, lcb_matrix, sorted_list, partition):
        ranked_list = []
        for _, k in enumerate(sorted_list):
            cur_p = list(partition[k])

            if self.rank == "random":
                np.random.shuffle(cur_p)
            elif self.rank == "certain":
                parent = {}
                child = {}
                for m in cur_p:
                    for n in cur_p:
                        if lcb_matrix[m][n] > 0.5:
                            if m not in child.keys():
                                child[m] = [n]
                            else:
                                child[m].append(n)
                            if n not in parent.keys():
                                parent[n] = [m]
                            else:
                                parent[n].append(m)
                # topological sort
                candidate = []
                for m in cur_p:
                    if m not in parent.keys():
                        candidate.append(m)

                ranked_id = []
                while len(candidate) != 0:
                    node = np.random.choice(candidate)
                    ranked_id.append(node)
                    candidate.remove(node)
                    if node in child.keys():
                        children = child[node]
                    else:
                        children = []
                    for j in children:
                        parent[j].remove(node)
                        if len(parent[j]) == 0:
                            candidate.append(j)
                cur_p = ranked_id
            else:
                print("Rank method is not supported.")
                sys.exit()

            ranked_list.extend(cur_p)

        return np.array(ranked_list)

    def _create_train_ranking(self, query_id, query_feat, inverted):
        lcb_matrix = self.get_lcb(query_feat)
        partition, sorted_list, certain_edges = self.get_partitions(lcb_matrix)
        self._last_query_feat = query_feat
        self.ranking = self.get_ranking(lcb_matrix, sorted_list, partition)

        return self.ranking

    def update_to_interaction(self, clicks):
        if np.any(clicks):
            self._update_to_clicks(clicks)

    def generate_pairs(self, clicks):
        n_docs = self.ranking.shape[0]
        cur_k = np.minimum(n_docs, self.n_results)
        included = np.ones(cur_k, dtype=np.int32)
        if not clicks[-1]:
            included[1:] = np.cumsum(clicks[::-1])[:0:-1]
        neg_ind = np.where(np.logical_xor(clicks, included))[0]
        pos_ind = np.where(clicks)[0]

        if self.ind:
            np.random.shuffle(neg_ind)
            np.random.shuffle(pos_ind)
            pairs = list(zip(pos_ind, neg_ind))
        else:
            pairs = list(itertools.product(pos_ind, neg_ind))

        return np.array(pairs)

    def _update_to_clicks(self, clicks):
        # generate all pairs from the clicks
        pairs = self.generate_pairs(clicks)
        # update the model if we have valid observed pairs
        if len(pairs) != 0:
            # update covariance matrix A
            for p in pairs:
                pos_doc_idx = self.ranking[p[0]]
                neg_doc_idx = self.ranking[p[1]]
                diff_feat = (self._last_query_feat[pos_doc_idx] - self._last_query_feat[neg_doc_idx]).reshape(1, -1)
                self.InvA -= multi_dot([self.InvA, diff_feat.T, diff_feat, self.InvA]) / float(
                    1 + np.dot(np.dot(diff_feat, self.InvA), diff_feat.T))  # update history and update model
            self.update_history_data(pairs)
            self.update_to_history()

    def update_history_data(self, pairs):
        query_index = self.get_query_global_index(self._last_query_id, self._train_query_ranges)
        query_pairs = self.ranking[pairs] + query_index
        if self.doc_pair_history is not None:
            self.doc_pair_history = np.append(self.doc_pair_history, query_pairs, 0)
        else:
            self.doc_pair_history = query_pairs

        # idx = len(self.history)
        # self.history[idx] = {}
        # self.history[idx]["qid"] = self._last_query_id
        # self.history[idx]["pairs"] = query_pairs
        self.n_data += len(query_pairs)

    def generate_training_data(self):
        train_x = []
        train_y = []

        # for idx in self.history.keys():
        # for i in range(len(self.doc_pair_history)):
        #     pair = self.doc_pair_history[i]
        # pos_ids = [pair[0] for pair in pairs]
        # neg_ids = [pair[1] for pair in pairs]
        # x = self._train_features[pos_ids] - self._train_features[neg_ids]
        # train_x.append(x)
        # y = np.ones(len(pairs))
        # train_y.append(y)
        pos_ids = self.doc_pair_history[:, 0]
        neg_ids = self.doc_pair_history[:, 1]
        train_x = self._train_features[pos_ids] - self._train_features[neg_ids]
        train_y = np.ones(len(train_x))

        # train_x = np.vstack(train_x)
        # train_y = np.hstack(train_y)

        return train_x, train_y

    def update_to_history(self):
        train_x, train_y = self.generate_training_data()
        myargs = (train_x, train_y)
        betas = np.random.rand(train_x.shape[1])
        result = minimize(self.cost_func_reg, x0=betas, args=myargs, method="L-BFGS-B", jac=self.log_gradient_reg,
                          options={"ftol": 1e-6}, )
        self.model.update_weights(result.x)
