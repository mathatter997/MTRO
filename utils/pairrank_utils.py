import math
import numpy as np
from itertools import islice


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def logist(s):
    return 1 / (1 + np.exp(-s))


def logistic_func(theta, x):
    return float(1) / (1 + math.e ** (-x.dot(theta)))


def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))


def update_edges(super_edges, candidate_nodes, new_id):
    # create new edges for the new node
    new_edges = set()
    for i in candidate_nodes:
        new_edges = new_edges.union(super_edges[i])
    new_edges.remove(new_id)
    super_edges[new_id] = new_edges
    # remove the merged nodes
    for i in candidate_nodes:
        if i != new_id:
            super_edges.pop(i)
    # update the edges in other nodes
    for i in super_edges.keys():
        if super_edges[i] & candidate_nodes:
            for n in candidate_nodes:
                try:
                    super_edges[i].remove(n)
                except Exception as e:
                    # print(e)
                    pass
            if i != new_id:
                super_edges[i].add(new_id)
    return super_edges


def update_nodes(super_nodes, candidate_nodes, new_id):
    new_node = set()
    for i in candidate_nodes:
        new_node = new_node.union(set(super_nodes[i]))
    super_nodes[new_id] = new_node
    for i in candidate_nodes:
        if i != new_id:
            super_nodes[i] = {}
    return super_nodes
