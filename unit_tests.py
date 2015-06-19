import unittest
import gt_tools
import pd_tools
import printing
from graph_tool.all import *
import numpy as np


class TestGTTools(unittest.TestCase):
    def setUp(self):
        pass

    def test_net_from_adj(self):
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
