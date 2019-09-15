import gudhi
import numpy as np
from . import data_processing
import linkcom
import networkx as nx

# agent_handler -> numpy array
# array history includes processed screen input


def find_patterns(data: np.array):
    """
    :param data: a TxN binary matrix
    """
    patterns = {}
    for i in range(len(data.shape[0])):
        indices = tuple(np.nonzero(data[i]))
        if indices in patterns:
            patterns[indices] += 1
        else:
            patterns[indices] = 1
    return patterns


def build_gudhi_tree(patterns):
    tree = gudhi.SimplexTree()
    for simplex, frequency in patterns.items():
        tree.insert(simplex, 1.0 / frequency)
    return tree


class CliqueComplex:
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method="avg", max_edge_length=10,
                 tree_dimension=None):
        """
        :param data: a TxN matrix where T is the time dim and N is the variable
        dimension.
        """
        self._data = data
        self._preprocessed_data = data_processing.bin_time_series(
                data_processing.pre_process_layer_states(self._data, filter_lowest),
                window_size, method)
        self._distance_matrix = np.abs(np.corrcoef(self._preprocessed_data,
                                                   rowvar=False))
        self.rips_complex = gudhi.RipsComplex(distance_matrix=self._distance_matrix,
                                              max_edge_length=max_edge_length)
        if tree_dimension is None:
            self.simplex_tree = self.rips_complex.create_simplex_tree()
        else:
            self.simplex_tree = self.rips_complex.create_simplex_tree(tree_dimension)


class ConcurrenceComplex:
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method="avg", threshold=0.5):
        """
        :param data: a TxN matrix where T is the time dim and N is the variable
        dimension.
        """
        self._data = data
        self._preprocessed_data = data_processing.binarize_time_series(
            data_processing.bin_time_series(
                data_processing.pre_process_layer_states(self._data, filter_lowest),
                    window_size, method), threshold)
        self.simplex_tree = build_gudhi_tree(find_patterns(self._preprocessed_data))


def convert_edge_membership_to_node_membership(n, edge_membership):
    number_of_communities = len(set(edge_membership.values()))
    node_membership_matrix = np.zeros((number_of_communities, n), dtype=bool)
    for edge, membership in edge_membership.items():
        node_membership_matrix[edge[0], membership] = True
        node_membership_matrix[edge[1], membership] = True

    return node_membership_matrix


class IndependenceComplex:
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method='avg'):
        self._data = data
        self._preprocessed_data = data_processing.bin_time_series(
                data_processing.pre_process_layer_states(self._data, filter_lowest),
                window_size, method)
        self._correlation_matrix = np.abs(np.corrcoef(self._preprocessed_data,
                                                      rowvar=False))
        np.fill_diagonal(self._correlation_matrix, 0)
        # TODO: look into determining going from correlation -> graph
        # particularly concerning significance/thresholding

        g = nx.from_numpy_matrix(self._correlation_matrix,
                                 created_using=nx.DiGraph())
        # keys=edges and values=community membership, best similarity,
        # best partition density
        edge_membership, best_sim, best_part = linkcom.cluster(g, is_weighted=True)
        self._node_membership = convert_edge_membership_to_node_membership(
            len(g), edge_membership)
        self._non_membership = ~self._node_membership
        self.simplex_tree = build_gudhi_tree(find_patterns(self._non_membership))
