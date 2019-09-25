__all__ = ['find_patterns', 'build_gudhi_tree', 'VRComplex',
           'ConcurrenceComplex', 'convert_edge_membership_to_node_membership',
           'IndependenceComplex']


import gudhi
import numpy as np
from collections import Counter
from tdavis.data_processing import *
import linkcom
import networkx as nx


def find_patterns(data: np.array, verbose=False):
    """
    :param data: a TxN binary matrix
    """
    patterns = {}
    for i in range(data.shape[0]):
        indices = tuple(np.nonzero(data[i])[0].tolist())
        if indices in patterns:
            patterns[indices] += 1
        else:
            patterns[indices] = 1
    if verbose:
        print("Pattern distribution")
        print("\tNum patterns:", len(patterns))
        print("\tPattern frequency:")
        for pattern, freq in patterns.items():
            print(freq, len(pattern))
    return patterns


def build_gudhi_tree(patterns):
    print("building tree...")
    tree = gudhi.SimplexTree()
    for simplex, frequency in patterns.items():
        tree.insert(simplex, 1.0 / frequency)
    print("tree built...")
    return tree


class Complex:
    def __init__(self, simplex_tree, **kwargs):
        """
        :param homology_coeff_field: has to be prime int.
        """
        self.simplex_tree = simplex_tree
        print("calculating persistence...")
        self.persistence = self.simplex_tree.persistence(
            kwargs.get('homology_coeff_field', 11),
            kwargs.get('min_persistence', 0.0))

    def get_betti(self):
        return [Counter([p[0] for p in self.persistence])[i]
                for i in range(self.simplex_tree.dimension())]

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


class VRComplex(Complex):
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method="avg", max_edge_length=10,
                 tree_dimension=None, **kwargs):
        """
        :param data: a TxN matrix where T is the time dim and N is the variable
        dimension.
        """
        self._data = data
        self._preprocessed_data = bin_time_series(
                pre_process_layer_states(self._data, filter_lowest),
                window_size, method)
        self._distance_matrix = 1 - np.abs(np.corrcoef(self._preprocessed_data,
                                                       rowvar=False))
        self.rips_complex = gudhi.RipsComplex(distance_matrix=self._distance_matrix,
                                              max_edge_length=max_edge_length)
        if tree_dimension is None:
            simplex_tree = self.rips_complex.create_simplex_tree()
        else:
            simplex_tree = self.rips_complex.create_simplex_tree(tree_dimension)
        print("Complex of dimension", simplex_tree.dimension(),
              'Simplices', simplex_tree.num_simplices(),
              'Vertices', simplex_tree.num_vertices())
        super().__init__(simplex_tree, **kwargs)


# class ConcurrenceComplex(Complex):
#     def __init__(self, data: np.array, filter_lowest=0.2,
#                  window_size=5, method="avg", threshold=0.5, **kwargs):
#         """
#         :param data: a TxN matrix where T is the time dim and N is the variable
#         dimension.
#         """
#         self._data = data
#         self._preprocessed_data = binarize_time_series(
#             bin_time_series(
#                 pre_process_layer_states(self._data, filter_lowest),
#                     window_size, method), threshold)
#         super().__init__(build_gudhi_tree(find_patterns(self._preprocessed_data,
#                                                         kwargs.get("verbose", False))),
#                          **kwargs)

class ConcurrenceComplex(Complex):
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method="max", threshold=0.5, **kwargs):
        """
        :param data: a TxN matrix where T is the time dim and N is the variable
        dimension.
        """
        self._data = data
        self._preprocessed_data = bin_time_series(
            max_binary(self._data), window_size, method)
        super().__init__(build_gudhi_tree(find_patterns(self._preprocessed_data,
                                                        kwargs.get("verbose", False))),
                         **kwargs)


def convert_edge_membership_to_node_membership(graph, edge_membership,
                                               min_com_size=4):
    """
    For linkcom, nodes will have membership in each community for which it
    shares an edge.
    :param graph: nx graph
    :param edge_membership: dictionary of edge->community
    :param min_com_size: minimum # of edges needed to make a community.
    :return: CxN matrix
    """
    community_frequencies = Counter(edge_membership.values())
    communities = [com for com, freq in community_frequencies.items()
                   if freq > min_com_size]
    print("Num non-trivial communities", len(communities))
    print("Community distribution")
    for com, freq in community_frequencies.items():
        if freq > min_com_size:
            print("com", com, "freq", freq)

    community_index = {com: i for i, com in enumerate(communities)}
    node_membership_matrix = np.zeros((len(communities),
                                       nx.number_of_nodes(graph)),
                                      dtype=bool)
    for edge, membership in edge_membership.items():
        if membership in community_index:
            node_membership_matrix[community_index[membership],
                                   edge[0]] = True
            node_membership_matrix[community_index[membership],
                                   edge[1]] = True

    return node_membership_matrix


def convert_edge_membership_to_edge_member_matrix(graph, edge_membership):
    """
    :param graph: nx graph
    :param edge_membership: dictionary of edge->community
    :return: CxE matrix
    """
    community_frequencies = Counter(edge_membership.values())
    communities = [com for com, freq in community_frequencies.items()
                   if freq > 1]
    community_index = {com: i for i, com in enumerate(communities)}
    edge_membership_matrix = np.zeros((len(communities),
                                       nx.number_of_edges(graph)),
                                      dtype=bool)
    for i, edge in enumerate(graph.edges()):
        if edge_membership[edge] in community_index:
            edge_membership_matrix[community_index[edge_membership[edge]], i] = True

    return edge_membership_matrix


class IndependenceComplex(Complex):
    def __init__(self, data: np.array, filter_lowest=0.2,
                 window_size=5, method='avg', threshold=0.0, **kwargs):
        self._data = data
        self._preprocessed_data = bin_time_series(
                pre_process_layer_states(self._data, filter_lowest),
                window_size, method)
        self._matrix = np.abs(np.corrcoef(self._preprocessed_data,
                                          rowvar=False))
        np.fill_diagonal(self._matrix, 0)
        self._matrix[self._matrix < threshold] = 0
        self.graph = nx.from_numpy_matrix(self._matrix)
        print("num nodes:", nx.number_of_nodes(self.graph),
              "num edges:", nx.number_of_edges(self.graph))
        # keys=edges and values=community membership
        cluster_results = linkcom.cluster(self.graph, is_weighted=True)
        self.edge_membership = cluster_results[0]
        print('edge members', self.edge_membership)
        self._node_membership = convert_edge_membership_to_node_membership(
            self.graph, self.edge_membership)
        self._non_membership = ~self._node_membership
        self.write_gexf_to_file("test")
        pattern_plot(self._non_membership, xlabel="com", ylabel="node")
        super().__init__(build_gudhi_tree(find_patterns(self._non_membership)),
                         **kwargs)

    def write_gexf_to_file(self, filename):
        community_frequencies = Counter(self.edge_membership.values())
        communities = [com for com, freq in community_frequencies.items()
                       if freq > 1]
        community_index = {com: i for i, com in enumerate(communities)}
        edge_comunities = {edge: {'community': community_index[self.edge_membership[edge]]}
                           if self.edge_membership[edge] in community_index
                           else {'community': -1}
                           for edge in self.graph.edges()}
        nx.set_edge_attributes(self.graph, edge_comunities)
        nx.write_gexf(self.graph, filename + ".gexf")
