from __future__ import division
from sys import platform as _platform
import matplotlib
import matplotlib.cm as colormap

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from graph_tool.all import *
import os
import matplotlib.cm as colormap
from matplotlib.colors import ColorConverter as color_converter
import pandas as pd
import Image
import ImageDraw
import ImageFont
import subprocess
import printing
import random
import datetime
import copy
import shutil
import numpy as np
import operator
import math
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import scipy.stats as stats
from scipy.sparse import csr_matrix, issparse
import sys
from scipy.stats import powerlaw, poisson
from collections import defaultdict
import traceback
from basics import create_folder_structure
import powerlaw as fit_powerlaw


def print_f(*args, **kwargs):
    if 'class_name' not in kwargs:
        kwargs.update({'class_name': 'gt_tools'})
    printing.print_f(*args, **kwargs)


def load_edge_list(filename, directed=False, id_dtype='int'):
    store_fname = filename + '.gt'
    if os.path.isfile(store_fname):
        try:
            g = load_graph(store_fname)
        except:
            print 'failed loading. recreate graph'
            os.remove(store_fname)
            return load_edge_list(filename)
        if 'mtime' in g.gp.keys() and g.gp['mtime'] != os.path.getmtime(filename):
            print 'modified edge-list. recreate graph'
            os.remove(filename + '.gt')
            return load_edge_list(filename)
        else:
            return g
    else:
        g = Graph(directed=directed)
        nodeid_to_v = defaultdict(g.add_vertex)
        with open(filename, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    nodes = map(int, line.strip().split())
                    try:
                        src = nodeid_to_v[nodes[0]]
                    except IndexError:
                        continue
                    dest = map(lambda x: nodeid_to_v[x], nodes[1:])
                    for d in dest:
                        g.add_edge(src, d)
        node_id_pmap = g.new_vertex_property(id_dtype)
        for id, v in nodeid_to_v.iteritems():
            node_id_pmap[v] = id
        g.vp['NodeId'] = node_id_pmap
        g.gp['filename'] = g.new_graph_property('string')
        g.gp['filename'] = filename
        g.gp['mtime'] = g.new_graph_property('object', os.path.getmtime(filename))
        g.save(filename + '.gt', fmt='gt')
    return g


def net_from_adj(mat, directed=True, parallel_edges=True):
    g = Graph(directed=directed)
    assert mat.shape[0] == mat.shape[1]
    if not issparse(mat):
        mat = csr_matrix(mat)
    g.add_vertex(mat.shape[0])
    if not parallel_edges:
        if np.issubdtype(mat.dtype, int):
            w = g.new_edge_property('int')
        else:
            w = g.new_edge_property('float')
    row_idx, col_idx = mat.nonzero()
    for d, r, c in zip(mat.data, row_idx, col_idx):
        src_v = g.vertex(c)
        dest_v = g.vertex(r)
        if parallel_edges:
            for i in xrange(int(d)):
                g.add_edge(src_v, dest_v)
        else:
            w[g.add_edge(src_v, dest_v)] = d
    if not parallel_edges:
        g.ep['weights'] = w
    return g


def load_property(network, filename, type='int', resolve='NodeId', sep=None, line_groups=False):
    assert isinstance(network,Graph)
    type = type.lower()
    if type == 'str' or type == 'string':
        pmap = network.new_vertex_property('string')
        mapper = lambda x: str(x)
    else:
        pmap = network.new_vertex_property(type)
        if type == 'int':
            mapper = lambda x: int(x)
        else:
            mapper = lambda x: float(x)
    if resolve is not None:
        res_map = network.vp[resolve]
        resolve = {str(res_map[v]): v for v in network.vertices()}
    mapped_vertices = set()
    com_id = 0
    with open(filename, 'r') as f:
        for line in filter(lambda l_line: not l_line.startswith('#'), f):
            line = line.strip().split(sep)
            if line_groups:
                line_vertices = []
                appender = line_vertices.append
                for i in line:
                    try:
                        if resolve is None:
                            i = network.vertex(int(i))
                        else:
                            i = resolve[i]
                        appender(i)
                    except KeyError:
                        pass
                for v in line_vertices:
                    pmap[v] = mapper(com_id)
                    mapped_vertices.add(int(v))
                com_id += 1
            else:
                try:
                    v = network.vertex(int(line[0])) if resolve is None else resolve[line[0]]
                    pmap[v] = mapper(line[1])
                    mapped_vertices.add(int(v))
                except KeyError:
                    pass
    unmapped_v = set(map(int,network.vertices())) - mapped_vertices
    if unmapped_v:
        print filename, 'contained no mapping for', len(unmapped_v) / network.num_vertices() * 100, '% of all vertices'
        print 'unmapped vertices:', list(unmapped_v)[:100]
    return pmap


class SBMGenerator():
    @staticmethod
    def gen_stock_blockmodel(num_nodes=100, blocks=3, self_con=1, other_con=0.2, directed=False, degree_seq='powerlaw',
                             powerlaw_exp=2.1, num_links=300, loops=False):
        g = Graph(directed=directed)
        com_pmap = g.new_vertex_property('int')
        nodes_range = np.array(range(num_nodes))
        for idx in nodes_range:
            com_pmap[g.add_vertex()] = idx % blocks
        g.vp['com'] = com_pmap
        other_con /= ((blocks - 1) if blocks > 1 else 1)

        prob_pmap = g.new_vertex_property('float')
        block_to_vertices = dict()
        block_to_cumsum = dict()
        if degree_seq == 'powerlaw':
            degree_seq = (1 - stats.powerlaw.rvs(powerlaw_exp, size=num_nodes))
        elif degree_seq == 'random':
            degree_seq = np.random.random(size=num_nodes)
        elif degree_seq == 'exp':
            degree_seq = np.random.exponential(size=num_nodes)
        else:
            degree_seq = np.array([1.0] * num_nodes)
        degree_seq.sort()
        # print degree_seq
        block_deg_seq_sum = dict()
        for i in range(blocks):
            vertices_in_block = list(filter(lambda x: com_pmap[x] == i, g.vertices()))
            block_to_vertices[i] = vertices_in_block
            block_deg_seq = degree_seq[nodes_range % blocks == i]
            deg_seq_sum = block_deg_seq.sum()
            block_deg_seq /= deg_seq_sum
            cum_sum = np.cumsum(block_deg_seq)
            assert np.allclose(cum_sum[-1], 1)
            block_to_cumsum[i] = cum_sum
            block_deg_seq_sum[i] = deg_seq_sum
            for v, p in zip(vertices_in_block, block_deg_seq):
                prob_pmap[v] = p
        blocks_prob = list()
        for i in range(blocks):
            row = list()
            for j in range(blocks):
                if i == j:
                    val = self_con
                else:
                    val = other_con
                row.append(val * block_deg_seq_sum[i] * block_deg_seq_sum[j])
            blocks_prob.append(np.array(row))
        blocks_prob = np.array(blocks_prob)
        blocks_prob /= blocks_prob.sum()
        # print blocks_prob
        cum_sum = np.cumsum(blocks_prob)
        assert np.allclose(cum_sum[-1], 1)
        # print cum_sum
        links_created = 0
        for v in g.vertices():
            if True or v.in_degree() + v.out_degree() == 0:
                src_block = com_pmap[v]
                while True:
                    dest_b = SBMGenerator.get_one_random_block(cum_sum, blocks, src_block)
                    dest_v = block_to_vertices[dest_b][SBMGenerator.get_random_node(block_to_cumsum[dest_b])]
                    if loops or v != dest_v and g.edge(v, dest_v) is None:
                        g.add_edge(v, dest_v)
                        links_created += 1
                        break

        for link_idx in range(num_links - links_created):
            edge = 1
            src_v, dest_v = None, None
            while edge is not None:
                src_b, dest_b = SBMGenerator.get_random_blocks(cum_sum, blocks)
                src_v = block_to_vertices[src_b][SBMGenerator.get_random_node(block_to_cumsum[src_b])]
                dest_v = block_to_vertices[dest_b][SBMGenerator.get_random_node(block_to_cumsum[dest_b])]
                edge = g.edge(src_v, dest_v)
                if edge is None and not loops and src_v == dest_v:
                    edge = 1
            g.add_edge(src_v, dest_v)
        return g

    @staticmethod
    def get_random_node(cum_sum):
        rand_num = np.random.random()
        idx = 0
        for idx, i in enumerate(cum_sum):
            if i >= rand_num:
                return idx
        print 'warn: get rand num till end'
        return idx

    @staticmethod
    def get_random_blocks(cum_sum, num_blocks):
        rand_num = np.random.random()
        idx = 0
        for idx, i in enumerate(cum_sum):
            if i >= rand_num:
                return idx % num_blocks, int(idx / num_blocks)
        print 'warn: get rand block till end'
        return idx % num_blocks, int(idx / num_blocks)

    @staticmethod
    def get_one_random_block(cum_sum, num_blocks, row):
        src_b = None
        dest_b = None
        while src_b is None or row != src_b:
            src_b, dest_b = SBMGenerator.get_random_blocks(cum_sum, num_blocks)
        return dest_b

    @staticmethod
    def analyse_graph(g, filename='output/net', draw_net=False):
        print str(g)
        deg_map = g.degree_property_map('total')
        plt.close('all')
        ser = pd.Series(deg_map.a)
        ser.plot(kind='hist', bins=int(deg_map.a.max()), lw=0)
        plt.xlabel('degree')
        plt.ylabel('num nodes')
        res = fit_powerlaw.Fit(deg_map.a)
        print 'powerlaw alpha:', res.power_law.alpha
        print 'powerlaw xmin:', res.power_law.xmin
        plt.title('powerlaw alpha:' + str(res.power_law.alpha) + ' || powerlaw xmin:' + str(res.power_law.xmin))
        plt.savefig(filename + '_degdist.png', bbox_tight=True)
        plt.close('all')
        if draw_net:
            graph_draw(g, vertex_fill_color=g.vp['com'], output_size=(200, 200),
                       vertex_size=prop_to_size(deg_map, mi=2, ma=15, power=1.), output=filename + '_network.png',
                       bg_color=[1, 1, 1, 1])


# Generator Class works with GraphTool generators, as they provide more functionality than NetworkX Generators
class GraphGenerator():
    # init generator
    def __init__(self, num_nodes=5000, directed=False):
        self.directed = directed
        self.num_nodes = num_nodes
        self.graph = None
        self.node_id_map = None
        self.return_and_reset()

    def return_and_reset(self):
        result = self.graph
        self.graph = Graph(directed=self.directed)
        self.node_id_map = defaultdict(lambda x: self.graph.add_vertex())
        return result

    # start creating blockmodel graph
    def create_blockmodel_graph(self, blocks=7, connectivity=10, model="blockmodel-traditional"):
        def corr(a, b):
            if a == b:
                return 0.999
            else:
                return 0.001

        self.print_f("Starting to create Blockmodel Graph with {} nodes and {} blocks".format(self.num_nodes, blocks))

        self.graph, vertex_colors = random_graph(self.num_nodes, lambda: poisson(connectivity), directed=False, model=model, block_membership=lambda: random.randint(1, blocks),
                                                 vertex_corr=corr)
        self.graph.vertex_properties["colorsComm"] = vertex_colors
        return self.return_and_reset()

    def create_fully_connected_graph(self, size=1000, directed=False, self_edges=False):
        return self.create_stochastic_blockmodel_graph(blocks=1, size=size, directed=directed, self_edges=self_edges, self_block_connectivity=1.0, other_block_connectivity=1.0)

    @staticmethod
    def create_sbm_lined_up_matrix(blocks=10, self_block_connectivity=None, other_block_connectivity=None):
        if self_block_connectivity is None:
            self_block_connectivity = [0.9]
        elif isinstance(self_block_connectivity, (int, float)):
            self_block_connectivity = [self_block_connectivity]
        if other_block_connectivity is None:
            other_block_connectivity = [0.1]
        elif isinstance(other_block_connectivity, (int, float)):
            other_block_connectivity = [other_block_connectivity]
        connectivity_matrix = []
        blocks_range = range(blocks)
        for idx in blocks_range:
            row = []
            outer_prob = other_block_connectivity[idx % len(other_block_connectivity)]
            inner_prob = self_block_connectivity[idx % len(self_block_connectivity)]
            for jdx in blocks_range:
                if idx != jdx:
                    row.append(outer_prob / pow(abs(idx - jdx), 2))
                else:
                    row.append(inner_prob)
            connectivity_matrix.append(row)
        return connectivity_matrix

    # scale = None
    # scale = relative
    # scale = absolute
    def create_stochastic_blockmodel_graph(self, blocks=10, size=100, self_block_connectivity=0.9, other_block_connectivity=0.1, connectivity_matrix=None, directed=False,
                                           self_edges=False, power_exp=None, scale=None, plot_stat=False):
        size = size if isinstance(size, list) else [size]
        self_block_connectivity = self_block_connectivity if isinstance(self_block_connectivity, list) else [self_block_connectivity]
        other_block_connectivity = other_block_connectivity if isinstance(other_block_connectivity, list) else [other_block_connectivity]

        num_nodes = sum([size[i % len(size)] for i in xrange(blocks)])
        if power_exp is None:
            self.print_f("Starting to create Stochastic Blockmodel Graph with {} nodes and {} blocks".format(num_nodes, blocks))
        else:
            self.print_f("Starting to create degree-corrected (alpha=" + str(power_exp) + ") Stochastic Blockmodel Graph with {} nodes and {} blocks".format(num_nodes, blocks))
        self.print_f('convert/transform probabilities')
        blocks_range = range(blocks)
        block_sizes = np.array([size[i % len(size)] for i in blocks_range])

        # create connectivity matrix of self- and other-block-connectivity
        if connectivity_matrix is None:
            connectivity_matrix = []
            self.print_f('inner conn: ' + str(self_block_connectivity) + '\tother conn: ' + str(other_block_connectivity))
            for idx in blocks_range:
                row = []
                for jdx in blocks_range:
                    if idx == jdx:
                        row.append(self_block_connectivity[idx % len(self_block_connectivity)])
                    else:
                        if scale is not None:
                            prob = other_block_connectivity[idx % len(other_block_connectivity)] / (num_nodes - block_sizes[idx]) * block_sizes[jdx]
                            if directed:
                                row.append(prob)
                            else:
                                row.append(prob / 2)
                        else:
                            row.append(other_block_connectivity[idx % len(other_block_connectivity)])
                connectivity_matrix.append(row)

        # convert con-matrix to np.array
        if connectivity_matrix is not None and isinstance(connectivity_matrix, np.matrix):
            connectivity_matrix = np.asarray(connectivity_matrix)

        # convert con-matrix to np.array
        if connectivity_matrix is not None and not isinstance(connectivity_matrix, np.ndarray):
            connectivity_matrix = np.array(connectivity_matrix)

        self.print_f('conn mat')
        printing.print_matrix(connectivity_matrix)

        if scale == 'relative' or scale == 'absolute':
            new_connectivity_matrix = []
            for i in blocks_range:
                connectivity_row = connectivity_matrix[i, :] if connectivity_matrix is not None else None
                nodes_in_src_block = block_sizes[i]
                multp = 1 if scale == 'absolute' else (nodes_in_src_block * (nodes_in_src_block - 1))
                row_prob = [(connectivity_row[idx] * multp) / (nodes_in_src_block * (nodes_in_block - 1)) for idx, nodes_in_block in enumerate(block_sizes)]
                new_connectivity_matrix.append(np.array(row_prob))
            connectivity_matrix = np.array(new_connectivity_matrix)
            self.print_f(scale + ' scaled conn mat:')
            printing.print_matrix(connectivity_matrix)

        # create nodes and store corresponding block-id
        self.print_f('insert nodes')
        vertex_to_block = []
        appender = vertex_to_block.append
        colors = self.graph.new_vertex_property("float")
        for i in xrange(blocks):
            block_size = size[i % len(size)]
            for j in xrange(block_size):
                appender((self.graph.add_vertex(), i))
                node = vertex_to_block[-1][0]
                colors[node] = i

        # create edges
        get_rand = np.random.random
        add_edge = self.graph.add_edge

        self.print_f('create edge probs')
        degree_probs = defaultdict(lambda: dict())
        for vertex, block_id in vertex_to_block:
            if power_exp is None:
                degree_probs[block_id][vertex] = 1
            else:
                degree_probs[block_id][vertex] = math.exp(power_exp * np.random.random())

        tmp = dict()
        self.print_f('normalize edge probs')
        all_prop = []
        for block_id, node_to_prop in degree_probs.iteritems():
            sum_of_block_norm = 1 / sum(node_to_prop.values())
            tmp[block_id] = {key: val * sum_of_block_norm for key, val in node_to_prop.iteritems()}
            all_prop.append(tmp[block_id].values())
        degree_probs = tmp
        if plot_stat:
            plt.clf()
            plt.hist(all_prop, bins=15)
            plt.savefig("prop_dist.png", bbox_tight=True)
            plt.close('all')

        self.print_f('count edges between blocks')
        edges_between_blocks = defaultdict(lambda: defaultdict(int))
        for idx, (src_node, src_block) in enumerate(vertex_to_block):
            conn_mat_row = connectivity_matrix[src_block, :]
            for dest_node, dest_block in vertex_to_block:
                if get_rand() < conn_mat_row[dest_block]:
                    edges_between_blocks[src_block][dest_block] += 1

        self.print_f('create edges')
        for src_block, dest_dict in edges_between_blocks.iteritems():
            self.print_f(' -- Processing Block {}. Creating links to: {}'.format(src_block, dest_dict))
            for dest_block, num_edges in dest_dict.iteritems():
                self.print_f('   ++ adding {} edges to {}'.format(num_edges, dest_block))
                for i in xrange(num_edges):
                    # find src node
                    prob = np.random.random()
                    prob_sum = 0
                    src_node = None
                    for vertex, v_prob in degree_probs[src_block].iteritems():
                        prob_sum += v_prob
                        if prob_sum >= prob:
                            src_node = vertex
                            break
                    # find dest node
                    prob = np.random.random()
                    prob_sum = 0
                    dest_node = None
                    for vertex, v_prob in degree_probs[dest_block].iteritems():
                        prob_sum += v_prob
                        if prob_sum >= prob:
                            dest_node = vertex
                            break
                    if src_node is None or dest_node is None:
                        print 'Error selecting node:', src_node, dest_node
                    if self.graph.edge(src_node, dest_node) is None:
                        if self_edges or not src_node == dest_node:
                            add_edge(src_node, dest_node)
        self.graph.vertex_properties["colorsComm"] = colors
        return self.return_and_reset()

    def create_preferential_attachment(self, communities=10):
        self.graph = price_network(self.num_nodes, directed=False, c=0, gamma=1, m=1)
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 1000, communities)
        return self.return_and_reset()

    # add node to graph and check if node is in node_dict
    def add_node(self, node_id, further_mappings=None):
        v = self.node_id_map[node_id]
        self.graph.vp['NodeId'][v] = node_id
        if further_mappings is not None:
            assert isinstance(further_mappings, dict)
            for key, val in further_mappings.iteritems():
                self.graph.vp[key][v] = val
        return v

    def load_smw_collab_network(self, filename, communities=10):
        self.print_f("Creating Graph")
        id_prop = self.graph.new_vertex_property("int")
        self.graph.vp["label"] = self.graph.new_vertex_property("string")

        f = open(filename, "rb")
        for idx, line in enumerate(f):
            if idx % 1000 == 0:
                self.print_f("--> parsing line %d" % idx)
            split_line = line.strip("\n").split("\t")
            source_v = self.add_node(split_line[0], id_prop)
            if split_line[1] != "":
                target_v = self.add_node(split_line[1], id_prop)
                self.graph.add_edge(source_v, target_v)

        self.print_f("Detecting Communities")
        self.graph.vp['colorsComm'] = community_structure(self.graph, 1000, communities)
        remove_self_loops(self.graph)
        remove_parallel_edges(self.graph)
        return self.return_and_reset()

    @staticmethod
    def increment_neighbours(vertices, b):
        for n in vertices:
            b[int(n)] += 1

    # start creating random graph
    # NOTE:
    # If min_degree is too small, graph will be disconnected and consist of many smaller graphs!
    # This could make diffusion problematic!
    def create_random_graph(self, min_degree=2, max_degree=40, model="probabilistic", communities=10):
        # Function to sample edges between nodes!
        self.print_f('create random graph')

        def sample_k(min_val, max_val, k=None):
            accept = False
            while not accept:
                k = random.randint(min_val, max_val + 1)
                accept = random.random() < 1.0 / k
            return k

        self.graph = random_graph(self.num_nodes, lambda: sample_k(min_degree, max_degree), model=model, vertex_corr=lambda i, k: 1.0 / (1 + abs(i - k)), directed=self.directed,
                                  n_iter=100)
        self.graph.vp['colorsComm'] = community_structure(self.graph, 10000, max_degree / communities)
        return self.return_and_reset()

    # start loading  graph
    def create_karate_graph(self):
        self.graph = collection.data["karate"]
        # Removing descriptions and readme, as they screw with the GML parser of networkx!
        self.graph.gp['description'] = ''
        self.graph.gp['readme'] = ''
        # Calculating Colors and updating members
        self.graph.vp['colorsComm'] = community_structure(self.graph, 10000, 2)
        self.directed = self.graph.is_directed()
        self.num_nodes = self.graph.num_vertices()
        return self.return_and_reset()

    def loaded_post_action(self):
        self.directed = self.graph.is_directed()
        self.num_nodes = self.graph.num_vertices()
        self.print_f("Graph loaded with {} nodes and {} edges".format(self.graph.num_vertices(), self.graph.num_edges()))

    # load graph from gml
    def load_gml(self, fn):
        self.print_f("Loading GML")
        self.graph = load_graph(fn)
        self.loaded_post_action()
        return self.return_and_reset()

    # load graph from file
    def load_gt(self, fn):
        self.print_f("Loading GT")
        self.graph = load_graph(fn)
        self.loaded_post_action()
        return self.return_and_reset()

    @staticmethod
    def print_f(*args, **kwargs):
        kwargs.update({'class_name': 'GraphGenerator'})
        print_f(*args, **kwargs)


def calc_eigenvalues(graph, num_ev=100):
    num_ev = min(100, num_ev)
    print_f("Extracting adjacency matrix!")
    adj_mat = adjacency(graph, weight=None)
    print_f("Starting calculation of {} Eigenvalues".format(num_ev))
    evals_large_sparse, evecs_large_sparse = largest_eigsh(adj_mat, num_ev * 2, which='LM')
    print_f("Finished calculating Eigenvalues")
    weights = sorted([float(x) for x in evals_large_sparse], reverse=True)[:num_ev]
    graph.gp["top_eigenvalues"] = graph.new_graph_property("vector<float>", weights)
    return graph


def cleanup_graph(graph, largest_comp=True, parallel_edges=False, self_loops=False):
    if largest_comp:
        reduce_to_largest_component(graph)
    if not parallel_edges:
        remove_parallel_edges(graph)
    if not self_loops:
        remove_self_loops(graph)
    return graph


def reduce_to_largest_component(graph):
    print_f("Reducing graph to largest connected component!")
    l = label_largest_component(graph)
    graph = GraphView(graph, vfilt=l)
    graph.purge_vertices(in_place=True)
    return graph


def calc_vertex_properties(graph, max_iter_ev=1000, max_iter_hits=1000):
    print_f("Calculating PageRank")
    graph.vp["pagerank"] = pagerank(graph)

    print_f("Calculating Clustering Coefficient")
    graph.vp["clustercoeff"] = local_clustering(graph)

    print_f("Calculating Eigenvector Centrality")
    ev, ev_centrality = eigenvector(graph, weight=None, max_iter=max_iter_ev)
    graph.vp["evcentrality"] = ev_centrality

    print_f("Calculating HITS")
    eig, authorities, hubs = hits(graph, weight=None, max_iter=max_iter_hits)
    graph.vp["authorities"] = authorities
    graph.vp["hubs"] = hubs

    print_f("Calculating Degree Property Map")
    graph.vertex_properties["degree"] = graph.degree_property_map("total")
    return graph
