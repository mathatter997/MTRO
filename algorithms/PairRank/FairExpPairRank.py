import numpy as np
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.PairRank.PairRank import PairRank
from utils.fair_utils import generate_all_combination, position_probability
import copy


class FairExpPairRank(PairRank):
    def __init__(self, alpha, _lambda, refine, rank, update, learning_rate, learning_rate_decay, ind, decay_mode,
                 unfairness, fair_alpha, fair_epsilon, *args, **kargs):
        super().__init__(alpha, _lambda, refine, rank, update, learning_rate, learning_rate_decay, ind, *args, **kargs)

        # for fairness
        self.fair_alpha = fair_alpha
        self.fair_epsilon = fair_epsilon
        self.decay_mode = decay_mode
        self.unfairness = unfairness

        self.uf_dict = {}
        self.uf = {}
        self.group_candidates = {self.n_results: generate_all_combination(self.n_results)}
        self.create_unfairness_table(self.n_results)
        self.current_unfairness = 0
        self.violation = 0

    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({"learning_rate": 0.1, "learning_rate_decay": 1.0})
        return parent_parameters

    def create_unfairness_table(self, k):
        self.uf_dict[k] = {}
        position_bias = position_probability(k, self.decay_mode)
        uf_table = self.group_candidates[k] * (-1 - self.fair_alpha) + 1
        self.uf[k] = np.dot(uf_table, position_bias)
        for i in range(len(self.group_candidates[k])):
            group_sequence = self.group_candidates[k][i]
            self.uf_dict[k]["".join(str(v) for v in group_sequence)] = self.uf[k][i]

    def get_unfairness(self, group_sequence):
        position_bias = position_probability(self.n_results, self.decay_mode)
        uf = group_sequence * (-1 - self.fair_alpha) + 1
        return np.dot(uf, position_bias[: len(group_sequence)])

    def get_candidate_group_sequence(self, k):
        next_unfairness = self.current_unfairness + self.uf[k]
        sorted_index = np.argsort(np.abs(next_unfairness))
        sorted_unfairness = next_unfairness[sorted_index]

        idx = np.where(np.abs(sorted_unfairness) > self.fair_epsilon)[0][0]

        return sorted_index, idx

    @staticmethod
    def rank_block(block_links, child, query_group, k):
        group_links = {}
        ranking = {}
        for group in [0, 1]:
            group_links[group] = [v.copy() for v in block_links if query_group[v[0]] == group]

            index = {}
            ranking[group] = []
            for i, doc_links in enumerate(group_links[group]):
                index[doc_links[0]] = i

            while len(ranking[group]) < min(k, len(group_links[group])):
                sorted_block = sorted(group_links[group], key=lambda x: (x[1], x[2]))
                doc_links = sorted_block[0]
                doc = doc_links[0]
                ranking[group].append(doc)
                group_links[group][index[doc]][1] = 100000
                for child_doc in child[doc]:
                    if child_doc in index.keys():
                        idx = index[child_doc]
                        if idx < len(group_links[group]):
                            group_links[group][idx][1] -= 1
        return ranking

    def generate_ranking_with_template(self, group_sequence, query_group, child, doc_links, blocks, sorted_list):
        ranking = []
        sid = 0
        for i, m in enumerate(sorted_list):
            block_size = len(list(blocks[m]))
            eid = min(sid + block_size, self.n_results)

            block_group_sequence = group_sequence[sid:eid]
            seq_group = {0: eid - sid - sum(block_group_sequence), 1: sum(block_group_sequence)}

            block_group = query_group[list(blocks[m])]
            blk_group = {1: sum(block_group), 0: block_size - sum(block_group)}
            num_missing = 0

            if blk_group[0] < seq_group[0] or blk_group[1] < seq_group[1]:
                missing_group = 0
                if blk_group[1] < seq_group[1]:
                    missing_group = 1
                num_missing = seq_group[missing_group] - blk_group[missing_group]

                promoted_doc = []
                for j in range(i + 1, len(sorted_list)):

                    block_id = sorted_list[j]
                    target_b = list(blocks[block_id])
                    target_index = np.where(query_group[target_b] == missing_group)[0]
                    target_docs = [target_b[n] for n in target_index]
                    block_links = [doc_links[n] for n in target_docs]
                    target_ranking = self.rank_block(block_links, child, query_group,
                                                     min(num_missing, len(block_links)))

                    promoted_doc.extend(target_ranking[missing_group])
                    num_missing -= len(target_ranking[missing_group])
                    for doc in target_ranking[missing_group]:
                        blocks[block_id].remove(doc)
                    if num_missing == 0:
                        break

                for doc in promoted_doc:
                    blocks[m].add(doc)

            if num_missing != 0:
                break

            block_links = [doc_links[i] for i in blocks[m]]
            block_ranking = self.rank_block(block_links, child, query_group, len(block_group_sequence))
            for g in block_group_sequence:
                doc = block_ranking[g][0]
                ranking.append(doc)
                block_ranking[g].remove(doc)
                blocks[m].remove(doc)

            if len(ranking) == len(group_sequence):
                break

            if i + 1 < len(sorted_list):
                next_block = sorted_list[i + 1]
                for doc in blocks[m]:
                    blocks[next_block].add(doc)
                sid = eid
            else:
                break

        return ranking

    def find_best_ranking(self, lcb_matrix, blocks, sorted_list, certain_edges, query_group, sorted_index, idx, k):
        n_doc = len(lcb_matrix)
        doc_links = []
        child = {}
        parent = {}
        for i in range(n_doc):
            doc_links.append([i, 0, 0])
            child[i] = []
            parent[i] = []
        for edge in certain_edges:
            s, e = edge
            doc_links[s][2] -= 1
            doc_links[e][1] += 1
            child[s].append(e)
            parent[e].append(s)

        candidate_ranking = []
        num_misorders = []
        for gid in sorted_index[:idx]:
            group_sequence = self.group_candidates[k][gid]
            block_tmp = copy.deepcopy(blocks)
            sl_tmp = sorted_list[:]
            if sum(query_group) >= sum(group_sequence) and len(query_group) - sum(query_group) >= len(
                    group_sequence) - sum(group_sequence):
                ranking = self.generate_ranking_with_template(group_sequence, query_group, child, doc_links, block_tmp,
                                                              sl_tmp)

                # count the number of violations for the satisfied sequence
                if len(ranking) == len(group_sequence):
                    count = 0
                    for m in range(k):
                        for n in range(m + 1, k):
                            if (ranking[n], ranking[m]) in certain_edges:
                                count += 1
                    candidate_ranking.append(ranking)
                    num_misorders.append(count)
                    if count == 0:
                        break

        if len(num_misorders) == 0:
            # print("some query does not fit")
            ranking = None
            for gid in sorted_index[idx:]:
                group_sequence = self.group_candidates[k][gid]
                if sum(query_group) >= sum(group_sequence) and len(query_group) - sum(query_group) >= len(
                        group_sequence) - sum(group_sequence):
                    ranking = self.generate_ranking_with_template(group_sequence, query_group, child, doc_links, blocks,
                                                                  sorted_list)
                    if len(ranking) == len(group_sequence):
                        break  # else:
            result_ranking = ranking
            count = 0
            for m in range(k):
                for n in range(m + 1, k):
                    if (result_ranking[n], result_ranking[m]) in certain_edges:
                        count += 1
            nm = count
        else:
            if num_misorders[-1] == 0:
                result_ranking = candidate_ranking[-1]
                nm = 0
            else:
                idx = np.argmin(np.array(num_misorders))
                result_ranking = candidate_ranking[idx]
                nm = num_misorders[idx]
        return result_ranking, nm

    def _create_train_ranking(self, query_id, query_feat, inverted):
        k = min(self.n_results, len(query_feat))
        if k not in self.uf.keys():
            self.group_candidates[k] = generate_all_combination(k)
            self.create_unfairness_table(k)

        sorted_index, idx = self.get_candidate_group_sequence(k)
        lcb_matrix = self.get_lcb(query_feat)
        blocks, sorted_list, certain_edges = self.get_partitions(lcb_matrix)
        query_group = self.get_query_groups(self._last_query_id, self._train_groups, self._train_query_ranges)
        self.partition = blocks
        self.sorted_list = sorted_list
        ranking, num_violation = self.find_best_ranking(lcb_matrix, blocks, sorted_list, certain_edges, query_group,
                                                        sorted_index, idx, k)

        self.ranking = ranking
        self.violation = num_violation

        self._last_query_feat = query_feat
        self.ranking = np.array(self.ranking)

        return self.ranking

    def update_to_interaction(self, clicks):
        query_group = self.get_query_groups(self._last_query_id, self._train_groups, self._train_query_ranges)
        if self.unfairness == 'projected':
            k = min(len(self.ranking), self.n_results)
            ranking_group = query_group[self.ranking]
            self.current_unfairness += self.uf_dict[k]["".join(str(v) for v in ranking_group)]
        else:
            print("unfairness calculation is not included")
            exit()

        if np.any(clicks):
            self._update_to_clicks(clicks)
