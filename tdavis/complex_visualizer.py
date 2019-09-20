__all__ = ['find_patterns', 'build_gudhi_tree', 'CliqueComplex',
           'ConcurrenceComplex', 'convert_edge_membership_to_node_membership',
           'IndependenceComplex']


import gudhi
import numpy as np
import matplotlib.pyplot as plt
from tdavis.data_processing import *
import linkcom
import networkx as nx


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


class Complex:
    def __init__(self, simplex_tree, homology_coeff_field=11):
        """
        :param homology_coeff_field: has to be prime int.
        """
        self.simplex_tree = simplex_tree
        self.persistence = self.simplex_tree.persistence(homology_coeff_field)

    def plot_betti_distribution(self, filtration_range=None):
        if filtration_range is None:
            betti_dist = self.simplex_tree.persistent_betti_numbers(
                filtration_range[0], filtration_range[1])
        else:
            betti_dist = self.simplex_tree.betti_numbers()

    def plot_persistence_diagram(self, *args, **kwargs):
        ax = gudhi.plot_persistence_diagram(self.persistence, *args, **kwargs)

    # TODO: add distribution for # max simplices for given filtration
    # check some papers to see if anyone even uses ones like these, may
    # just not do it, or just do community detection on the distance matrix

class CliqueComplex(Complex):
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method="avg", max_edge_length=10,
                 tree_dimension=None):
        """
        :param data: a TxN matrix where T is the time dim and N is the variable
        dimension.
        """
        self._data = data
        self._preprocessed_data = bin_time_series(
                pre_process_layer_states(self._data, filter_lowest),
                window_size, method)
        self._distance_matrix = np.abs(np.corrcoef(self._preprocessed_data,
                                                   rowvar=False))
        self.rips_complex = gudhi.RipsComplex(distance_matrix=self._distance_matrix,
                                              max_edge_length=max_edge_length)
        if tree_dimension is None:
            simplex_tree = self.rips_complex.create_simplex_tree()
        else:
            simplex_tree = self.rips_complex.create_simplex_tree(tree_dimension)
        super().__init__(simplex_tree)


class ConcurrenceComplex(Complex):
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method="avg", threshold=0.5):
        """
        :param data: a TxN matrix where T is the time dim and N is the variable
        dimension.
        """
        self._data = data
        self._preprocessed_data = binarize_time_series(
            bin_time_series(
                pre_process_layer_states(self._data, filter_lowest),
                    window_size, method), threshold)
        super().__init__(build_gudhi_tree(find_patterns(self._preprocessed_data)))


def convert_edge_membership_to_node_membership(n, edge_membership):
    number_of_communities = len(set(edge_membership.values()))
    node_membership_matrix = np.zeros((number_of_communities, n), dtype=bool)
    for edge, membership in edge_membership.items():
        node_membership_matrix[edge[0], membership] = True
        node_membership_matrix[edge[1], membership] = True

    return node_membership_matrix


class IndependenceComplex(Complex):
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method='avg'):
        self._data = data
        self._preprocessed_data = bin_time_series(
                pre_process_layer_states(self._data, filter_lowest),
                window_size, method)
        self._correlation_matrix = np.abs(np.corrcoef(self._preprocessed_data,
                                                      rowvar=False))
        np.fill_diagonal(self._correlation_matrix, 0)
        # TODO: look into determining going from correlation -> graph
        # particularly concerning significance/thresholding

        self.graph = nx.from_numpy_matrix(self._correlation_matrix,
                                          created_using=nx.DiGraph())
        # keys=edges and values=community membership
        edge_membership, _, _ = linkcom.cluster(g, is_weighted=True)
        self._node_membership = convert_edge_membership_to_node_membership(
            len(g), edge_membership)
        self._non_membership = ~self._node_membership
        super().__init__(build_gudhi_tree(find_patterns(self._non_membership)))

    def write_gexf_to_file(self, filename):
        # assign communities to nodes (not sure how gephi could handle it)
        # assign communities to edges
        # assign out-membership to nodes
