from __future__ import division, print_function
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
from . import printing
# import printing
import random
import datetime
import copy
import shutil
import numpy as np
import operator
import math
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import scipy.stats as stats
from scipy.sparse import csr_matrix, issparse, lil_matrix, dok_matrix
import sys
from scipy.stats import powerlaw, poisson
from collections import defaultdict
import traceback
from .basics import create_folder_structure
import powerlaw as fit_powerlaw



def print_f(*args, **kwargs):
    if 'class_name' not in kwargs:
        kwargs.update({'class_name': 'gt_tools'})
    printing.print_f(*args, **kwargs)


def add_vertex_property(g, fn, p_name='weight', vertex_id_to_vertex=None, vertex_id_dtype='int', property_dtype='int',
                        sep=None, comment='#', col=1):
    pmap = g.new_vertex_property(property_dtype)
    if property_dtype is 'int':
        property_type_mapper = int
    elif property_dtype is 'float':
        property_type_mapper = float
    else:
        property_type_mapper = str

    if vertex_id_dtype is 'int' or vertex_id_to_vertex is None:
        vertex_id_mapper = int
    elif vertex_id_dtype is 'float':
        vertex_id_mapper = float
    else:
        vertex_id_mapper = str
    get_vertex = g.vertex if vertex_id_to_vertex is None else (lambda x: vertex_id_to_vertex[x])
    with open(fn, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(comment):
                line = line.split(sep)
                v = get_vertex(vertex_id_mapper(line[0]))
                p = property_type_mapper(line[col])
                pmap[v] = p
    g.vp[p_name] = pmap


def load_edge_list(filename, directed=False, vertex_weights=None, edge_weights=None, vertex_id_name='NodeId',
                   vertex_id_dtype='int', sep=None, comment='#', multiple_edges_per_line=False, store=True,
                   try_load_gt=True):
    """
    Loads an edge-list into a Graph (graph-tool) object and returns the Graph object.

    Attributes
    ----------
    filename : str
        the filename of the edge-list
    directed : bool
        treat the edge-list directed (True) or not (False)
    vertex_weights : dict
        dict where keys are the name of the resulting property-map and values are dicts containing properties of the
        weight like data-type etc. (cf. add_vertex_property)
        important: dict containing the properties must contain 'filename' : "filename_of_mapping"
        Example:
            vertex_weights = dict()
            one_v_weight = dict()
            one_v_weight['filename'] = 'foo.file'
            one_v_weight['property_dtype'] = 'int'  # you can use anything available in add_vertex_property()
            vertex_weights['MyVertexWeight'] = one_v_weight

    edge_weights : dict
        dict where keys are the name of the resulting property_map and values are tuples containing column id where
        the weight is in the edge-list and the data-type.
        Example:
            weight_name : (0, 'int')
            creates an integer property-map containing the values found in the first column AFTER src, tar.
    vertex_id_name : str
        Name of property map where the original vertex-name/-id... should be stored
    vertex_id_dtype : str
        Data-type of the original vertex-name/-id
    sep : str or None
        separator which should be used to split the lines of the edge-list file
    comment : str
        lines starting with this string are treated as comments (ignored)
    multiple_edges_per_line : bool
        Do the lines contain multiple-edges. (e.g., src tar1 tar2 tar3)
        If this is true, edge_weights are not supported.
    store : bool
        Should the graph be stored in .gt format?
    try_load_gt : bool
        Should the function try to load a previously stored .gt file? ("filename.gt")
        Note if there where modifications in the original edge-list, the function will reload the edge-list.

    Notes
    -----
    -


    Examples
    --------
    -
     """
    store_fname = filename + '.gt'
    if os.path.isfile(store_fname) and try_load_gt:
        try:
            g = load_graph(store_fname)
        except:
            print('failed loading. recreate graph')
            os.remove(store_fname)
            return load_edge_list(filename, directed=directed, vertex_id_dtype=vertex_id_dtype, sep=sep,
                                  comment=comment)
        if 'mtime' in g.gp.keys() and os.path.isfile(filename) and g.gp['mtime'] != os.path.getmtime(filename):
            print('modified edge-list. recreate graph')
            os.remove(filename + '.gt')
            return load_edge_list(filename, directed=directed, vertex_id_dtype=vertex_id_dtype, sep=sep,
                                  comment=comment)
        else:
            return g
    else:
        g = Graph(directed=directed)
        nodeid_to_v = defaultdict(g.add_vertex)
        edge_list = []
        v_type = int

        edge_weights_dtypes = dict()
        edge_weights_pmaps = dict()
        edge_weights_names = dict()
        if edge_weights is not None:
            for name, (col, dtype) in edge_weights.iteritems():
                edge_weights_names[col] = name
                edge_weights_pmaps[col] = g.new_edge_property(dtype)
                if dtype == 'int':
                    edge_weights_dtypes[col] = int
                elif dtype == 'float':
                    edge_weights_dtypes[col] = float
                else:
                    edge_weights_dtypes[col] = float
        src_target_extractor = (
            lambda x: x[:2]) if edge_weights is not None or not multiple_edges_per_line else (
            lambda x: x)
        weights_extractor = (lambda x: None) if edge_weights is None else (
            lambda x: tuple([edge_weights_dtypes[idx](i) for
                             idx, i in enumerate(x[2:])]))
        edge_weights_list = list()
        with open(filename, 'r') as f:
            for line in filter(lambda x: not x.startswith(comment), map(lambda x: x.strip(), f)):
                line = line.split(sep)
                nodes = src_target_extractor(line)
                try:
                    nodes = map(v_type, nodes)
                except ValueError:
                    v_type = float
                    try:
                        nodes = map(v_type, nodes)
                    except ValueError:
                        v_type = str
                        nodes = map(v_type, nodes)
                try:
                    src = int(nodeid_to_v[nodes[0]])
                except IndexError:
                    continue
                dest = map(lambda x: int(nodeid_to_v[x]), nodes[1:])
                edges = [(src, d) for d in dest]
                edge_weights_list.append(weights_extractor(line))
                edge_list.extend(edges)
        g.add_edge_list(edge_list)
        if edge_weights is not None:
            edges = list(map(lambda x: g.edge(*x), edge_list))
            for col_idx, col_weights in enumerate(zip(*edge_weights_list)):
                current_pmap = edge_weights_pmaps[col_idx]
                for e, w in zip(edges, col_weights):
                    current_pmap[e] = w
                g.ep[edge_weights_names[col_idx]] = current_pmap
        if vertex_id_dtype is not None:
            node_id_pmap = g.new_vertex_property(vertex_id_dtype)
            for v_id, v in nodeid_to_v.iteritems():
                node_id_pmap[v] = v_id
            g.vp[vertex_id_name] = node_id_pmap
        g.gp['filename'] = g.new_graph_property('string', filename)
        g.gp['mtime'] = g.new_graph_property('object', os.path.getmtime(filename))

        if vertex_weights is not None:
            for name, props in vertex_weights.iteritems():
                kwargs = dict()
                kwargs['vertex_id_to_vertex'] = nodeid_to_v
                kwargs['vertex_id_dtype'] = vertex_id_dtype
                kwargs['sep'] = sep
                kwargs['comment'] = comment
                fn = props['filename']
                props.pop('filename', None)
                kwargs.update(props)
                add_vertex_property(g, fn, p_name=name, **kwargs)
        if store:
            g.save(filename + '.gt', fmt='gt')
    return g


def net_from_adj(mat, directed=True, parallel_edges=True):
    g = Graph(directed=directed)
    assert mat.shape[0] == mat.shape[1]
    if not issparse(mat):
        mat = csr_matrix(mat)
    elif not isinstance(mat, csr_matrix):
        mat = mat.tocsr()
    g.add_vertex(mat.shape[0])
    w = None
    if not parallel_edges:
        if np.issubdtype(mat.dtype, int):
            w = g.new_edge_property('int')
        else:
            w = g.new_edge_property('float')
    row_idx, col_idx = mat.nonzero()
    data = np.array(mat.data)
    if not directed:
        # diag-upper part only (including diag)
        row_idx, col_idx, data = zip(*[(r, c, d) for r, c, d in zip(row_idx, col_idx, data) if c <= r])
    if parallel_edges:
        row_idx = np.array([r for r, d in zip(row_idx, data) for i in range(int(d))])
        col_idx = np.array([c for c, d in zip(col_idx, data) for i in range(int(d))])
    g.add_edge_list(zip(col_idx, row_idx))
    if w:
        w.a = data
        g.ep['weights'] = w
    return g


def load_property(network, filename, type='int', resolve='NodeId', sep=None, line_groups=False):
    assert isinstance(network, Graph)
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
        print(filename, 'contained no mapping for', len(unmapped_v) / network.num_vertices() * 100, '% of all vertices')
        print('unmapped vertices:', list(unmapped_v)[:100])
    return pmap


def get_graph_com_connectivity(g, com_map='com'):
    com_map = g.vp[com_map]
    intern_edges = 0
    between_edges = 0
    for e in g.edges():
        if com_map[e.source()] == com_map[e.target()]:
            intern_edges += 1
        else:
            between_edges += 1
    num_edges = (intern_edges + between_edges)
    intern_edges /= num_edges
    between_edges /= num_edges
    return intern_edges, between_edges

def check_aperiodic(g):
    if isinstance(g, str):
        a = adjacency(load_graph(g))
        name = g.rsplit('/')[-1].replace('.gt', '')
        print('aperiodic:', name)
    else:
        a = adjacency(g)
    b = a * a
    diag_two_sum = b.diagonal().sum()
    print('\tA*A diag sum:', int(diag_two_sum))
    b *= a
    diag_three_sum = b.diagonal().sum()
    print('\tA*A*A diag sum:', int(diag_three_sum))
    aper = bool(diag_two_sum) and bool(diag_three_sum)
    print('\taperiodic:', aper)
    return aper

class SBMGenerator():
    @staticmethod
    def gen_stoch_blockmodel(num_nodes=1000, blocks=5, self_con=.97, other_con=0.03, directed=False,
                             degree_seq='powerlaw', powerlaw_exp=2.4, num_links=None, loops=False, min_degree=1,
                             con_prob_matrix=None, increase_lcc_prob=True, parallel_edges=False,
                             node_pick_strat=('dist', 'dist')):
        g = Graph(directed=directed)
        com_pmap = g.new_vertex_property('int')
        if isinstance(blocks, list):
            num_nodes = sum(blocks)
            num_blocks = len(blocks)
            block_sizes = blocks
        else:
            num_blocks = blocks
            block_sizes = [int(num_nodes / num_blocks) for i in range(num_blocks)]
            num_unmapped_nodes = num_nodes % num_blocks
            # print('unmapped nodes', num_unmapped_nodes)
            if num_unmapped_nodes > 0:
                for i in range(num_unmapped_nodes):
                    block_sizes[i] += 1
        g.add_vertex(num_nodes)

        if con_prob_matrix is not None:
            assert con_prob_matrix.shape[0] == con_prob_matrix.shape[1]
        com_pmap.a = np.hstack([np.array([idx] * i) for idx, i in enumerate(block_sizes)]).flatten()
        g.vp['com'] = com_pmap
        other_con /= ((num_blocks - 1) if num_blocks > 1 else 1)

        prob_pmap = g.new_vertex_property('float')
        block_to_vertices = dict()
        block_to_cumsum = dict()

        if isinstance(degree_seq, (np.ndarray, list, tuple)) and len(degree_seq) == num_nodes:
            degree_seq = degree_seq.astype('float')
            fixed_deg_seq = True
        else:
            fixed_deg_seq = False
            if degree_seq == 'powerlaw':
                degree_seq = stats.zipf.rvs(powerlaw_exp, loc=min_degree, size=num_nodes).astype('float')
            elif degree_seq == 'random':
                degree_seq = np.random.random(size=num_nodes)
            elif degree_seq == 'exp':
                degree_seq = np.random.exponential(size=num_nodes)
            else:
                degree_seq = np.array([1.0] * num_nodes)
            degree_seq.sort()
        degree_seq /= degree_seq.sum()
        multiplier = min_degree/degree_seq.min()
        degree_seq *= multiplier
        # print(degree_seq)
        if num_links is None:
            # print('min degree:', degree_seq.min())
            num_links = int(np.round((degree_seq.round().sum() / 2)))
            print('#links not set. using:', num_links)
        # print(degree_seq)
        block_deg_seq_sum = dict()
        vertices_array = np.array(list(map(int, g.vertices())))
        degree_indices = set(range(g.num_vertices()))
        for i in range(num_blocks):
            mask = com_pmap.a == i
            block_num_vertices = mask.sum()
            block_to_vertices[i] = vertices_array[mask]
            if fixed_deg_seq:
                block_deg_seq_idx = block_to_vertices[i]
            else:
                block_deg_seq_idx = random.sample(degree_indices, block_num_vertices)
                degree_indices -= set(block_deg_seq_idx)
            block_deg_seq = degree_seq[block_deg_seq_idx]
            deg_seq_sum = block_deg_seq.sum()
            block_deg_seq /= deg_seq_sum
            cum_sum = np.cumsum(block_deg_seq)
            assert np.allclose(cum_sum[-1], 1)
            block_to_cumsum[i] = cum_sum
            block_deg_seq_sum[i] = deg_seq_sum
            prob_pmap.a[mask] = block_deg_seq
        blocks_prob = list()
        for i in range(num_blocks):
            row = list()
            for j in range(num_blocks):
                if con_prob_matrix is None:
                    if i == j:
                        val = self_con
                    else:
                        val = other_con
                else:
                    val = con_prob_matrix.item((i, j))
                row.append(val * block_deg_seq_sum[i] * block_deg_seq_sum[j])
            blocks_prob.append(np.array(row))
        blocks_prob = np.array(blocks_prob)
        blocks_prob /= blocks_prob.sum()
        # print(blocks_prob)
        cum_sum = np.cumsum(blocks_prob)
        assert np.isclose(cum_sum[-1], 1)
        # print(cum_sum)
        if parallel_edges:
            edges = list()
            edges_adder = edges.append
        else:
            edges = set()
            edges_adder = edges.add

        if isinstance(node_pick_strat, str):
            node_pick_strat = (node_pick_strat, node_pick_strat)
        pick_funcs = dict()
        get_rnd_node = SBMGenerator.get_random_node
        inv_prob = SBMGenerator.inverse_prob
        pick_funcs['rnd'] = lambda b, s=None: random.choice(block_to_vertices[b])
        pick_funcs['dist'] = lambda b, s=None: block_to_vertices[b][get_rnd_node(block_to_cumsum[b])]
        pick_funcs['invdist'] = lambda b, s=None: block_to_vertices[b][
            get_rnd_node(inv_prob(block_to_cumsum[b]))]
        pick_funcs['dist_other_com_inv_dist'] = lambda b, s=None: \
            block_to_vertices[b][get_rnd_node(block_to_cumsum[b])] \
            if (s is None or s == b) else \
                block_to_vertices[b][get_rnd_node(inv_prob(block_to_cumsum[b]))]

        # more efficient way to pick both inverse to the dist
        if node_pick_strat[0] == node_pick_strat[1] == 'invdist':
            node_pick_strat[0] = node_pick_strat[1] = 'dist'
            block_to_cumsum = {key: inv_prob(val) for key, val in block_to_cumsum.items()}

        src_pick_func, dest_pick_func = map(lambda x: pick_funcs[x], node_pick_strat)

        get_one_rnd_block = SBMGenerator.get_one_random_block
        if increase_lcc_prob:
            for v in g.vertices():
                if directed or v.out_degree() == 0:
                    src_block = com_pmap[v]
                    init_len = len(edges)
                    while init_len == len(edges):
                        dest_b = get_one_rnd_block(cum_sum, num_blocks, src_block) \
                            if 'dist_other_com_inv_dist' not in node_pick_strat else src_block
                        dest_v = dest_pick_func(dest_b, src_block)
                        link = (int(v), dest_v)
                        is_loop = v == dest_v
                        if not is_loop:
                            if not directed:
                                link = tuple(sorted(link))
                            edges_adder(link)
                        elif loops:
                            edges_adder(link)

        get_rnd_blocks = SBMGenerator.get_random_blocks
        for link_idx in range(num_links - len(edges)):
            while True:
                #maybe switch to: get random node. identify block. get random dest-block.
                src_b, dest_b = get_rnd_blocks(cum_sum, num_blocks)
                src_v = src_pick_func(src_b, dest_b)
                dest_v = dest_pick_func(dest_b, src_b)
                link = (src_v, dest_v)
                is_loop = src_v == dest_v
                if not is_loop:
                    if not directed:
                        link = tuple(sorted(link))
                    edges_adder(link)
                    break
                elif loops:
                    edges_adder(link)
                    break
        if not isinstance(edges, list):
            edges = list(edges)
        g.add_edge_list(edges)
        return g

    @staticmethod
    def inverse_prob(p_cum_sum):
        p = np.hstack([p_cum_sum[0], np.ediff1d(p_cum_sum)])
        p = 1. / p
        p /= p.sum()
        return p.cumsum()

    @staticmethod
    def get_random_node(cum_sum):
        rand_num = np.random.random()
        idx = np.searchsorted(cum_sum, rand_num)
        if idx == len(cum_sum):
            idx -= 1
        return idx

    @staticmethod
    def get_random_blocks(cum_sum, num_blocks):
        rand_num = np.random.random()
        idx = np.searchsorted(cum_sum, rand_num)
        row = int(idx / num_blocks)
        col = idx % num_blocks
        #print(rand_num)
        #print(row, col)
        #print(cum_sum)
        return row, col

    @staticmethod
    def get_one_random_block(cum_sum, num_blocks, row):
        src_b = None
        dest_b = None
        while src_b is None or row != src_b:
            src_b, dest_b = SBMGenerator.get_random_blocks(cum_sum, num_blocks)
        return dest_b

    @staticmethod
    def gen_bow_tie_model(scc_size, out_size, in_size, con_prob_matrix=None, **kwargs):
        blocks = [scc_size, out_size, in_size]
        if con_prob_matrix is None:
            con_prob_matrix = np.zeros((3, 3))
            # in-comp
            con_prob_matrix[0, 0] = 0.001  # self
            con_prob_matrix[0, 1] = 0.1  # scc
            con_prob_matrix[0, 2] = 0.05  # out

            # scc
            con_prob_matrix[1, 1] = 5  # self
            con_prob_matrix[1, 2] = 0.1  # out

            con_prob_matrix[2, 2] = 0.01  # self
        np.set_printoptions(formatter=dict(float=lambda x: '%.5f' % x))
        print(con_prob_matrix / con_prob_matrix.sum())
        return SBMGenerator.gen_stoch_blockmodel(blocks=blocks, con_prob_matrix=con_prob_matrix,
                                                 directed=True, **kwargs)



    @staticmethod
    def analyse_graph(g, filename='output/net', draw_net=False):
        print(str(g))
        deg_map = g.degree_property_map('total')
        plt.close('all')
        ser = pd.Series(deg_map.a)
        ser.plot(kind='hist', bins=int(deg_map.a.max()), lw=0)
        plt.xlabel('degree')
        plt.ylabel('num nodes')
        res = fit_powerlaw.Fit(deg_map.a, discrete=True)
        print('powerlaw alpha:', res.power_law.alpha)
        print('powerlaw xmin:', res.power_law.xmin)
        plt.title('powerlaw alpha:' + str(res.power_law.alpha) + ' || powerlaw xmin:' + str(res.power_law.xmin))
        plt.savefig(filename + '_degdist.png', bbox_tight=True)
        plt.close('all')
        if draw_net:
            graph_draw(g, vertex_fill_color=g.vp['com'], output_size=(200, 200),
                       vertex_size=prop_to_size(deg_map, mi=2, ma=15, power=1.), output=filename + '_network.png',
                       bg_color=[1, 1, 1, 1])


def bow_tie(graph):
    assert graph.is_directed()
    largest_component = label_largest_component(graph)
    weakly_components = label_components(graph, directed=False)[0]
    largest_component_corresponding_weakly = list(
        set(weakly_components.a[np.nonzero(largest_component.a)[0]]))
    assert len(largest_component_corresponding_weakly) == 1
    largest_component_corresponding_weakly = largest_component_corresponding_weakly[0]
    wcc = (weakly_components.a == largest_component_corresponding_weakly)

    # Core, In and Out
    all_nodes = set(range(graph.num_vertices()))
    scc = set(np.nonzero(largest_component.a)[0])
    scc_node = random.sample(scc, 1)[0]
    graph_reversed = GraphView(graph, reversed=True)

    outc = np.nonzero(label_out_component(graph, scc_node).a)[0]
    inc = np.nonzero(label_out_component(graph_reversed, scc_node).a)[0]
    outc = set(outc) - scc
    inc = set(inc) - scc

    # Tubes, Tendrils and Other
    wcc = set(np.nonzero(wcc)[0])
    tube = set()
    out_tendril = set()
    in_tendril = set()
    other = all_nodes - wcc
    remainder = wcc - inc - outc - scc

    for idx, r in enumerate(remainder):
        print(idx + 1, '/', len(remainder), end='\r')
        predecessors = set(np.nonzero(label_out_component(graph_reversed, r).a)[0])
        successors = set(np.nonzero(label_out_component(graph, r).a)[0])
        if any(p in inc for p in predecessors):
            if any(s in outc for s in successors):
                tube.add(r)
            else:
                in_tendril.add(r)
        elif any(s in outc for s in successors):
            out_tendril.add(r)
        else:
            other.add(r)

    vp_bowtie = graph.new_vertex_property('string')
    for component, label in [
        (inc, 'IN'),
        (scc, 'SCC'),
        (outc, 'OUT'),
        (in_tendril, 'TL_IN'),
        (out_tendril, 'TL_OUT'),
        (tube, 'TUBE'),
        (other, 'OTHER')
    ]:
        for node in component:
            vp_bowtie[graph.vertex(node)] = label
    graph.vp['bowtie'] = vp_bowtie

    bow_tie = map(len, [inc, scc, outc, in_tendril, out_tendril, tube, other])
    assert sum(bow_tie) == graph.num_vertices()
    bow_tie = [100 * x / graph.num_vertices() for x in bow_tie]
    bow_tie = dict(IN=bow_tie[0], SCC=bow_tie[1], OUT=bow_tie[2], TL_IN=bow_tie[3], TL_OUT=bow_tie[4], TUBE=bow_tie[5],
                   OTHER=bow_tie[6])
    return bow_tie


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

        num_nodes = sum([size[i % len(size)] for i in range(blocks)])
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
        for i in range(blocks):
            block_size = size[i % len(size)]
            for j in range(block_size):
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
                for i in range(num_edges):
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
                        print('Error selecting node:', src_node, dest_node)
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

'''
def fast_sd(g, src_ids=None, dest_ids=None, pairs=None, max_dist=10, loops=False):
    # print(g)
    all_vertices_ids = list(map(int, g.vertices()))
    if src_ids is None:
        src_ids = all_vertices_ids
    elif not hasattr(src_ids, '__iter__'):
        src_ids = [int(src_ids)]
    if dest_ids is None:
        dest_ids = all_vertices_ids
    elif not hasattr(dest_ids, '__iter__'):
        dest_ids = [int(dest_ids)]
    if pairs is not None:
        if isinstance(pairs, list):
            pairs = set(pairs)
        elif isinstance(pairs, tuple):
            pairs = {pairs}
    elif g.is_directed():
        pairs = {(src, dest) for src in src_ids for dest in dest_ids if src != dest}
    else:
        pairs = {(src, dest) for src in src_ids for dest in dest_ids if src < dest}

    assert isinstance(pairs, set)

    A = adjacency(g).astype(bool)
    M = A.copy()
    M_old = None
    distances = defaultdict(lambda: defaultdict(int))
    m_max = g.num_vertices() * (g.num_vertices() - 1)
    reachable = 0
    for current_distance in range(1, max_dist + 1):
        m_non_zero = M.nnz

        if m_non_zero == 0:
            break
        reachable += m_non_zero
        print(current_distance, reachable/m_max)
        #pairs_with_current_distance = pairs & set(zip(*m_non_zero))
        #for src, dest in pairs_with_current_distance:
        #    distances[dest][src] = current_distance
        #pairs -= pairs_with_current_distance
        if not pairs:
            # print('break at:', current_distance)
            break
        M_old = M.copy()
        M *= A
        M -= M_old
        M = M.astype(bool)
    if len(pairs):
        # print('#pairs without distance:', len(pairs))
        pass
    return distances
'''
