from __future__ import division
from sys import platform as _platform
import matplotlib
import matplotlib.cm as colormap

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import sys

sys.path.append('./../')


import unittest
import gt_tools
import pd_tools
import printing
from graph_tool.all import *
import numpy as np
import os
import powerlaw as fit_powerlaw


class Test_net_from_adj(unittest.TestCase):
    def setUp(self):
        pass

    def test_unweighted_undirected_without_parallel(self):
        # unweighted undirected network without parallel edges
        g = price_network(100, m=50, directed=False)
        a = adjacency(g)
        g_restored = gt_tools.net_from_adj(a, directed=False)
        self.assertTrue(g.num_vertices() == g_restored.num_vertices())
        self.assertTrue(g.num_edges() == g_restored.num_edges())
        for e in g.edges():
            er = g_restored.edge(g_restored.vertex(e.source()), g_restored.vertex(e.target()))
            self.assertTrue(er is not None)
            self.assertTrue(int(er.source()) == int(e.source()))
            self.assertTrue(int(er.target()) == int(e.target()))

    def test_unweighted_directed_without_parallel(self):
        # unweighted directed network without parallel edges
        g = price_network(100, m=50)
        a = adjacency(g)
        g_restored = gt_tools.net_from_adj(a)
        self.assertTrue(g.num_vertices() == g_restored.num_vertices())
        self.assertTrue(g.num_edges() == g_restored.num_edges())
        for e in g.edges():
            er = g_restored.edge(g_restored.vertex(e.source()), g_restored.vertex(e.target()))
            self.assertTrue(er is not None)
            self.assertTrue(int(er.source()) == int(e.source()))
            self.assertTrue(int(er.target()) == int(e.target()))

    def test_unweighted_directed_with_parallel(self):
        # unweighted directed network with parallel edges
        g = price_network(100, m=50)
        for e in list(g.edges()):
            if np.random.random() > .5:
                for i in xrange(np.random.randint(1, 9)):
                    g.add_edge(e.source(), e.target())
        a = adjacency(g)
        g_restored = gt_tools.net_from_adj(a, parallel_edges=True)
        self.assertTrue(g.num_vertices() == g_restored.num_vertices())
        a_restored = adjacency(g_restored)
        r, c = a.nonzero()
        r_restored, c_restored = a_restored.nonzero()
        self.assertTrue(np.all(r == r_restored))
        self.assertTrue(np.all(c == c_restored))
        self.assertTrue(np.all(a.data == a_restored.data))

    def test_weighted_directed_without_parallel(self):
        # weighted directed network without parallel edges
        g = price_network(100, m=50)
        ew = g.new_edge_property('float')
        ew.a = np.random.random(g.num_edges())
        a = adjacency(g, weight=ew)
        g_restored = gt_tools.net_from_adj(a, parallel_edges=False)
        self.assertTrue(g.num_vertices() == g_restored.num_vertices())
        ew_restored = g_restored.ep['weights']
        for e in g.edges():
            er = g_restored.edge(g_restored.vertex(e.source()), g_restored.vertex(e.target()))
            self.assertTrue(er is not None)
            self.assertTrue(int(er.source()) == int(e.source()))
            self.assertTrue(int(er.target()) == int(e.target()))
            self.assertEqual(ew_restored[er], ew[e])

class Test_load_edge_list(unittest.TestCase):
    def setUp(self):
        pass

    def test_simple_edge_list(self):
        tmp_dir = './tmp/'
        try:
            os.mkdir(tmp_dir)
        except:
            pass
        seps = [' ', '\t', 'u']
        comments = '#'
        edges = set()
        for sep in seps:
            fn = tmp_dir + 'edge_list'
            with open(fn, 'w') as f:
                for s, t in zip(range(10, 20), reversed(range(10, 20))):
                    edges.add(tuple([s, t]))
                    f.write(str(s) + sep + str(t) + '\n')
                f.write('#comment')
            g = gt_tools.load_edge_list(fn, directed=True, sep=sep, comment=comments)
            os.remove(fn)
            os.remove(fn + '.gt')
            self.assertTrue('NodeId' in g.vp.keys())
            node_id_map = g.vp['NodeId']
            orig_node_ids = set(range(10, 20))
            reverse_map = dict()
            for v in g.vertices():
                orig_v = node_id_map[v]
                reverse_map[orig_v] = v
                self.assertTrue(orig_v in orig_node_ids)
            for e in g.edges():
                src = e.source()
                tar = e.target()
                edge_tuple = tuple([node_id_map[src], node_id_map[tar]])
                self.assertTrue(edge_tuple in edges)
            for src, tar in edges:
                src, tar = reverse_map[src], reverse_map[tar]
                self.assertIsNotNone(g.edge(src, tar))
        os.removedirs(tmp_dir)

    def test_weighted_edge_list(self):
        tmp_dir = './tmp/'
        try:
            os.mkdir(tmp_dir)
        except:
            pass
        seps = [' ', '\t', 'u']
        comments = '#'
        edges = set()
        for sep in seps:
            fn = tmp_dir + 'edge_list'
            with open(fn, 'w') as f:
                for s, t, w1, w2 in zip(range(10, 20), reversed(range(10, 20)), range(100, 110), np.arange(10) + 0.5):
                    edges.add(tuple([s, t]))
                    line_str = str(s) + sep + str(t) + sep + str(w1) + sep + str(w2) + '\n'
                    f.write(line_str)
                f.write('#comment')
            e_weights = dict()
            e_weights['firstw'] = (0, 'int')
            e_weights['secondw'] = (1, 'float')
            g = gt_tools.load_edge_list(fn, directed=True, sep=sep, comment=comments,edge_weights=e_weights)
            os.remove(fn)
            os.remove(fn + '.gt')
            self.assertTrue('NodeId' in g.vp.keys())
            self.assertTrue('firstw' in g.ep.keys())
            self.assertTrue('secondw' in g.ep.keys())
            node_id_map = g.vp['NodeId']
            f_pmap = g.ep['firstw']
            s_pmap = g.ep['secondw']
            self.assertTrue(np.all(np.array(f_pmap.a) == np.array(range(100, 110))))
            self.assertTrue(np.all(np.array(s_pmap.a) == np.arange(10) + 0.5))
            orig_node_ids = set(range(10, 20))
            reverse_map = dict()
            for v in g.vertices():
                orig_v = node_id_map[v]
                reverse_map[orig_v] = v
                self.assertTrue(orig_v in orig_node_ids)
            for e in g.edges():
                src = e.source()
                tar = e.target()
                edge_tuple = tuple([node_id_map[src], node_id_map[tar]])
                self.assertTrue(edge_tuple in edges)
            for src, tar in edges:
                src, tar = reverse_map[src], reverse_map[tar]
                self.assertIsNotNone(g.edge(src, tar))
        os.removedirs(tmp_dir)

    def test_node_weights_edge_list(self):
        tmp_dir = './tmp/'
        try:
            os.mkdir(tmp_dir)
        except:
            pass
        seps = [' ', '\t', 'u']
        comments = '#'
        edges = set()
        for sep in seps:
            fn = tmp_dir + 'edge_list'
            with open(fn, 'w') as f:
                for s, t, w1, w2 in zip(range(10, 20), reversed(range(10, 20)), range(100, 110), np.arange(10) + 0.5):
                    edges.add(tuple([s, t]))
                    line_str = str(s) + sep + str(t) + sep + str(w1) + sep + str(w2) + '\n'
                    f.write(line_str)
                f.write('#comment')
            fn_nw = tmp_dir + 'n_weights'
            with open(fn_nw, 'w') as f:
                f.write('\n'.join(map(lambda x: str(x[0]) + sep + str(x[1]), zip(range(10, 20), range(10)))))
            vertex_weights = dict()
            vertex_weights['weight'] = {'filename': fn_nw}
            g = gt_tools.load_edge_list(fn, directed=True, sep=sep, comment=comments, vertex_weights=vertex_weights)
            os.remove(fn)
            os.remove(fn_nw)
            os.remove(fn + '.gt')
            self.assertTrue('NodeId' in g.vp.keys())
            self.assertTrue('weight' in g.vp.keys())
            node_id_map = g.vp['NodeId']
            weight_map = g.vp['weight']
            orig_weights = np.array(range(10))
            for v in g.vertices():
                node_id = node_id_map[v]
                self.assertTrue(orig_weights[node_id - 10] == weight_map[v])
            orig_node_ids = set(range(10, 20))
            reverse_map = dict()
            for v in g.vertices():
                orig_v = node_id_map[v]
                reverse_map[orig_v] = v
                self.assertTrue(orig_v in orig_node_ids)
            for e in g.edges():
                src = e.source()
                tar = e.target()
                edge_tuple = tuple([node_id_map[src], node_id_map[tar]])
                self.assertTrue(edge_tuple in edges)
            for src, tar in edges:
                src, tar = reverse_map[src], reverse_map[tar]
                self.assertIsNotNone(g.edge(src, tar))
        os.removedirs(tmp_dir)

class Test_SBMGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = gt_tools.SBMGenerator()

    def test_simple_gen(self):
        self_con = .8
        other_con = 0.05
        g = self.gen.gen_stoch_blockmodel(min_degree=1, blocks=5, self_con=self_con, other_con=other_con,
                                          powerlaw_exp=2.1, degree_seq='powerlaw', num_nodes=1000)
        deg_hist = vertex_hist(g, 'total')
        res = fit_powerlaw.Fit(g.degree_property_map('total').a, discrete=True)
        print 'powerlaw alpha:', res.power_law.alpha
        print 'powerlaw xmin:', res.power_law.xmin
        if len(deg_hist[0]) != len(deg_hist[1]):
            deg_hist[1] = deg_hist[1][:len(deg_hist[0])]
        print 'plot degree dist'
        plt.plot(deg_hist[1], deg_hist[0])
        plt.xscale('log')
        plt.xlabel('degree')
        plt.ylabel('#nodes')
        plt.yscale('log')
        plt.savefig('deg_dist_test.png')
        plt.close('all')
        print 'plot graph'
        graph_draw(g, output='graph.png', output_size=(800, 800),
                   vertex_size=prop_to_size(g.degree_property_map('total'), mi=2, ma=30), vertex_color=[0., 0., 0., 1.],
                   vertex_fill_color=g.vp['com'],
                   bg_color=[1., 1., 1., 1.])
        plt.close('all')
        print 'init:', self_con / (self_con + other_con), other_con / (self_con + other_con)
        print 'real:', gt_tools.get_graph_com_connectivity(g, 'com')
