from __future__ import division
from graph_tool.all import *
import sys
import datetime

start = datetime.datetime.now()
filename = sys.argv[1]
assert filename.endswith('.gt')
g = load_graph(filename)
print 'calc pagerank'
g.vp['pagerank'] = pagerank(g)
print 'calc betweenness'
g.vp['betweenness'], g.ep['betweenness'] = betweenness(g)
print 'calc betweenness'
g.vp['betweenness'], g.ep['betweenness'] = betweenness(g)
print 'calc eigenvector'
eigenval, g.vp['eigenvector'] = eigenvector(g)
g.gp['eigenval'] = g.new_graph_property('float', eigenval)
print 'calc closeness'
g.vp['closeness'] = closeness(g)
print 'calc local clustering'
g.vp['local_clustering'] = local_clustering(g)
if len(sys.argv) > 2:
    g.save(sys.argv[2])
else:
    g.save(filename)
print 'all done', datetime.datetime.now() - start
