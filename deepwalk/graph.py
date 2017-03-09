"""
Graph utilities.

Originally by:
__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"
"""

import sys
import logging
from io import open
from os import path
from time import time
from glob import glob
import random
from random import shuffle

from six.moves import range, zip, zip_longest
from six import iterkeys

from collections import defaultdict, Iterable
from multiprocessing import cpu_count
from itertools import product, permutations

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool, cpu_count

logger = logging.getLogger("deepwalk")
logger.setLevel(logging.DEBUG)

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' Undirected graphs with self loops"""
    def __init__(self, is_directed=True, p=None, q=None):
        super(Graph, self).__init__(list)
        self.is_directed = is_directed
        if not is_directed:
            print('Graph is set to undirected, so you need to call '
                  'G.make_undirected() when you add nodes or edges.')
        
    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()
        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]
        return subgraph

    def make_undirected(self):
        t0 = time()
        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        logger.info('make_undirected: added missing edges {}s'.format(t1-t0))

        self.make_consistent()
        self.is_directed = False
        return self

    def make_consistent(self):
        "This makes the adjacency list a set."
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

        self.remove_self_loops()
        return self

    def remove_self_loops(self):
        removed = 0
        t0 = time()
        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1
        t1 = time()
        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True
        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, node):
        #if isinstance(nodes, Iterable):
        #    return {v:len(self[v]) for v in nodes}
        #else:
        return len(self[node])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.nodes()])/2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()

    def neighbors(self, node):
        "Returns the neighbors of a node"
        return list(set(self[node]))

    def edges(self):
        "Return list of edges"
        edges = []
        for x in self:
            for y in self[x]:
                edges.append((x, y))
        return edges

    def stringify(self):
        G = Graph()
        for node, nodes in self.items():
            G[str(node)] = [str(x) for x in nodes]
        return G

    def add_edge(self, src, dest):
        if src != dest:
            self[src].append(dest)
            if not self.is_directed:
                self[dest].append(src)
            else:
                raise Exception('is_directed == True but usually it should be False here.')

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(self.nodes())]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return path

    #------------------------------------------------------------------
    # node2vec integration begins

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        
        walks = []
        nodes = list(self.nodes())
        print 'Walk iteration:'
        for walk_iter in range(num_walks):
            print str(walk_iter+1), '/', str(num_walks)
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        #G = self
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(self.neighbors(dst)):
            if dst_nbr == src:
                #unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
                unnormalized_probs.append(1./p)
            elif self.has_edge(dst_nbr, src):
                #unnormalized_probs.append(G[dst][dst_nbr]['weight'])
                unnormalized_probs.append(1.)
            else:
                #unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
                unnormalized_probs.append(1./q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self, p=1, q=1):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        self.p = p
        self.q = q
        is_directed = self.is_directed

        alias_nodes = {}
        for node in self.nodes():
            #unnormalized_probs = [self[node][nbr]['weight'] for nbr in sorted(self.neighbors(node))]
            unnormalized_probs = [1. for nbr in sorted(self.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in self.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in self.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return self


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias samplinself.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend([row])

    return adjlist

def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

    if unchecked:
        parse_func = parse_adjacencylist_unchecked
        convert_func = from_adjlist_unchecked
    else:
        parse_func = parse_adjacencylist
        convert_func = from_adjlist

    adjlist = []

    t0 = time()

    with open(file_) as f:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            total = 0
            for idx, adj_chunk in enumerate(executor.map(parse_func, 
                                                         grouper(int(chunksize), f))):
                adjlist.extend(adj_chunk)
                total += len(adj_chunk)

    t1 = time()

    logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

    t0 = time()
    G = convert_func(adjlist)
    t1 = time()

    logger.info('Converted edges to graph in {}s'.format(t1-t0))

    if undirected:
        t0 = time()
        G = G.make_undirected()
        t1 = time()
        logger.info('Made graph undirected in {}s'.format(t1-t0))

    return G


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)

    G.make_consistent()
    return G


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    mat_matrix = mat_varables[variable_name]

    return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()
    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G

if __name__ == '__main__':

    G = Graph(p=1, q=1)
    G.preprocess_transition_probs()

    num_walks = 10
    walk_length = 80
    walks = G.simulate_walks(num_walks, walk_length)

