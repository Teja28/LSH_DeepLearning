import numpy as np
from numpy import linalg as LA
import collections

class SimpLSH:
    def __init__(self, num_hashes, num_tables,
                 dim, weights=None):
        self.num_hashes = num_hashes
        self.dim = dim
        self.num_tables = num_tables

        self.hyperplanes = [np.random.multivariate_normal(
            np.zeros(dim + 1), np.eye(dim + 1), num_hashes) for _ in range(L)]
        self.tables = []
        for _ in range(L):
            self.tables.append(collections.defaultdict(list))
        if weights is not None:
            assert(data.shape[0] == set_dim)
            self.weights = weights
            for i in range(self.weights.shape[1]):
                point = self.weights[:, 1]
                self.index(point)

    def hash(point, plane):
        projection = np.dot(plane, point)
        result = np.where(projection > 0, 1, 0)
        return result.dot(1 << np.arange(result.size)[::-1])

    def index(self, point):
        normed = point/np.sum(point)
        new_point = normed.append(sqrt(1 - LA.norm(normed, 2) ** 2))
        for i, table in enumerate(self.tables):
            signature = self.hash(new_point, self.hyperplanes[i])
            self.tables[i][signature].append(point)

    def query(self, point, distance, top_k=None):
        candidates = set()
        tf_query = point.append(0)
        for i, table in enumerate(self.tables):
            query_sig = self.hash(tf_query, self.hyperplanes[i])
            for key in table.keys():
                xor = query_sig ^ key
                hamming_dist = 0
                while xor > 0:
                    hamming_dist += xor & 1
                    xor >> 1
                if hamming_dist < distance:
                     for vec in table[key]:
                         candidates.add(table[key])

        d_func = SimpLSH.euclidian_dist
        candidates = [(ix, d_func(query_point, ix))
                      for ix in candidates]
        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k] if top_k else candidates


    @staticmethod
    def euclidian_dist(x, y):
        diff = x - y
        return np.sqrt(np.dot(diff, diff))
