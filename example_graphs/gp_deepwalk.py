import os
import json
import shutil
import datetime
from os.path import join, exists
import numpy as np
import pandas as pd
from collections import OrderedDict

from gensim.models import Word2Vec, KeyedVectors
from GPyOpt.methods import BayesianOptimization

from smm.wordvecs import run_gensim, run_fasttext, run_embedding
from deepwalk import graph
from scoring import load_blogcat, eval_blogcat

# This is the directory for a speciffic graph
base_dir = '/mnt/raid1/deepwalk/blogcat'

dirs = [base_dir, join(base_dir, 'walks'), join(base_dir, 'models'), join(base_dir, 'bo')]
for d in dirs:
    if not exists(d):
        os.makedirs(d)
    
d = join(base_dir, 'models')
if exists(d):
    shutil.rmtree(d)
    os.makedirs(d)
else:
    os.makedirs(d)

# Parameters in the parlance of gensim's Word2Vec
default_params = {
    'alpha': (0.025, float), # 0.05
    'size': (100, int),
    'window': (5, int),
    'iter': (5, int),
    'min_count': (0, int),
    'sample': (1e-3, float), # 1e-4
    # 'sg': (1, int), # genssim only
    # 'hs': (0, int), # gensim only
    'negative': (5, int),
    'loss': ('ns', str),
    'num_paths': (10, int),
    'path_length': (40, int),
    'workers': (12, int),
    'method': (1, int)
}

# parameters for deep walking
walk_keys = set(['num_paths', 'path_length'])

# parameters for optimizing
mixed_domain = [
    {'name': 'size', 'type': 'discrete', 'domain': (32, 64, 128)},
    {'name': 'method', 'type': 'discrete', 'domain': (0, 1)},
    {'name': 'window', 'type': 'discrete', 'domain': (5, 10, 15)},
    {'name': 'path_length', 'type': 'discrete', 'domain': (10, 40, 70)},
]

def vec2dict(X):
    # add parameters being optimized
    d = {}
    for i, dom in enumerate(mixed_domain):
        k = dom['name']
        typ = default_params[k][1]
        d[k] = typ(X[i])

    # add defaults for any parameters not being optimized
    for k, (val, typ) in default_params.items():
        if k not in d:
            d[k] = typ(val)
    return d

def run_from_params(params):
    walk_params = dict((k, params[k]) for k in walk_keys)
    walk_path = join(base_dir, 'walks', 'walks-%(num_paths)ix%(path_length)i-paths.txt' % walk_params)

    if exists(walk_path):
        #print('Walk path already exists.')
        pass
    else:
        #print('Walking...')
        walks = graph.build_deepwalk_corpus(G, **walk_params)

        #print('Writing walks...')
        with open(walk_path, 'w+') as f:
            f.write(' '.join([node for path in walks for node in path]))

    params['corpus_file'] = walk_path #"/mnt/raid1/deepwalk/blogcat/model-base.vec" #
    params['output_file'] = join(base_dir, 'models', 'model-%s.vec' % str(datetime.datetime.utcnow()))
    _ = run_embedding(params)

    res = eval_blogcat(params['output_file'],
                       G=G, 
                       labels_matrix=labels_matrix, 
                       training_percents=[0.6],
                       normalize=1, verbose=0)

    # negative b/c we're minimizing
    score = res[0.6][0]['micro']
    print 'micro-f1: %.3f' % score
    return -1.*score

class DwOpt:
    """
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """
    def __init__(self, input_dim, bounds=None, sd=None, run_id=None):
        self.input_dim = input_dim
        self.bounds = bounds
        self.evals = 0
        if sd == None: 
            self.sd = 0
        else: 
            self.sd = sd
        if run_id:
            self.fout = open(join(base_dir, 'log-%s.json' % run_id), 'w+')
        else:
            self.fout = None

    def f(self, X):
        X = np.reshape(X, self.input_dim)
        n = X.shape[0]
        assert n == len(mixed_domain)
        params = vec2dict(X)

        self.evals += 1
        params['eval'] = self.evals
        score = run_from_params(params)

        if self.fout is not None:
            params['score'] = score
            self.fout.write(json.dumps(params, indent=1) + '\n')
        print(params)
        return score

    def shutdown(self):
        if self.fout is not None:
            self.fout.close()


def save_opt(opt, run_id):
    outdir = join(base_dir, 'bo')
    opt.save_models(join(outdir, '%s-models.txt' % run_id))
    opt.save_evaluations(join(outdir, '%s-evals.txt' % run_id))
    opt.save_report(join(outdir, '%s-report.txt' % run_id))

    print opt.X
    print
    print opt.x_opt
    print 
    print opt.f(opt.x_opt)

if 0:
    import GPyOpt

    run_id = 'example'

    func = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=5) 
    
    mixed_domain =[{'name': 'var1_2', 'type': 'continuous', 'domain': (-10,10),'dimensionality': 2},
               {'name': 'var3', 'type': 'continuous', 'domain': (-8,3)},
               {'name': 'var4', 'type': 'discrete', 'domain': (-2,0,2)},
               {'name': 'var5', 'type': 'discrete', 'domain': (-1,5)}]

    myBopt = GPyOpt.methods.BayesianOptimization(f=func.f,                   # function to optimize       
                                                 domain=mixed_domain,        # box-constrains of the problem
                                                 initial_design_numdata = 20,# number data initial design
                                                 acquisition_type='EI',      # Expected Improvement
                                                 exact_feval = True)         # True evaluations

    max_iter = 10
    max_time = 60
    myBopt.run_optimization(max_iter, max_time)
    save_opt(myBopt, run_id)

if 1:
    
    run_id = 'dw1'

    G, labels_matrix = load_blogcat()

    input_dim = len(mixed_domain)
    func = DwOpt(input_dim, run_id=run_id)

    bo = BayesianOptimization(f=func.f,                  # function to optimize       
                             domain=mixed_domain,        # box-constrains of the problem
                             initial_design_numdata=20,  # number data initial design
                             acquisition_type='EI',      # Expected Improvement
                             exact_feval=True,           # True evaluations00
                             verbosity=True)

    max_iter = 20
    max_time = None
    bo.run_optimization(max_iter, max_time)
    save_opt(bo, run_id)
    func.shutdown()




