from itertools import permutations

import numpy as np


def generate_all_combination(k):
    result = []
    candidates = [0] * k
    result.append(candidates[:])

    for i in range(k):
        candidates[i] = 1
        perm = set(permutations(candidates, k))
        # print(list(perm))
        result.extend(list(perm))

    return np.array(result)


def position_probability(k, mode):
    prob = None
    if mode == "overk":
        prob = 1 / (np.arange(k) + 1)
    elif mode == "overlogk":
        prob = np.arange(k) + 1
        prob = 1 / (np.log2(prob) + 1)
    elif mode == "per":
        prob = np.array(
            [1.0, 0.99813462, 0.9924452, 0.98918079, 0.98896316, 0.98616509, 0.986134, 0.98436188, 0.98389554,
             0.98209234])[:k]
    elif mode == "nav":
        prob = np.array(
            [1.0, 0.69951133, 0.49562664, 0.35323587, 0.2482137, 0.17930765, 0.12760759, 0.09054698, 0.06689389,
             0.04597158])[:k]
    elif mode == "inf":
        prob = np.array(
            [1.0, 0.83557111, 0.6965989, 0.58099637, 0.49062832, 0.40953673, 0.34572045, 0.28968084, 0.24543319,
             0.2054151])[:k]
    else:
        print("Incorrect position decay mode")
        exit()
    return prob
