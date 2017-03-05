"""
Optimize DeepWalk parameters (walking and vectorizing) using GPyOpt (Gaussian Processes).
"""
import os
import sys
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
from deepwalk.scoring.scoring import load_blogcat, eval_blogcat

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# This is the directory for a speciffic graph
base_dir = '/mnt/raid1/deepwalk/blogcat'

# Stuff will get stored in all these subdirs
dirs = [base_dir, join(base_dir, 'walks'), 
        join(base_dir, 'models'), join(base_dir, 'bo')]
for d in dirs:
    if not exists(d):
        os.makedirs(d)
   
# Models folder grows fast so delete it every time 
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
    'sample': (1e-3, float), # 1e-4 threshold for configuring which higher-frequency words are randomly downsampled
    'sg': (1, int), # gensim onlyL if 1, skipgram is used, otherwise CBOW is used
    'hs': (0, int), # gensim only: if 1, hierarchical softmax will be used
    'negative': (5, int), # gensim & fasttext: use this many words for negative sampling
    'loss': ('ns', str),
    'num_paths': (10, int),
    'path_length': (40, int),
    'workers': (12, int),
    'method': (0, int),
    'maxn': (0, int), # fastText: 0 turns of char n-grams
}

# parameters for deep walking
walk_keys = set(['num_paths', 'path_length'])

# parameters for optimizing
mixed_domain = [
    {'name': 'size', 'type': 'discrete', 'domain': (32, 64, 100, 300, 512)},
    {'name': 'method', 'type': 'discrete', 'domain': (0, 1)},
    {'name': 'window', 'type': 'discrete', 'domain': (5, 10, 15, 20, 25)},
    #{'name': 'path_length', 'type': 'discrete', 'domain': (40, 70, 100, 130)},
    #{'name': 'num_paths', 'type': 'discrete', 'domain': (10, 20)},
]

# mixed_domain = [
#     {'name': 'size', 'type': 'discrete', 'domain': (64,)},
# ]

def run_from_params(params):
    walk_params = dict((k, params[k]) for k in walk_keys)

    # We load the walks if the current parameterization of them is already done.
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

    params['corpus_file'] = walk_path
    params['output_file'] = join(base_dir, 'models', 'model-%s.vec' % str(datetime.datetime.utcnow()))
    _ = run_embedding(params)

    res = eval_blogcat(params['output_file'],
                       G=G, 
                       labels_matrix=labels_matrix, 
                       training_percents=[0.6],
                       normalize=1, verbose=0)

    # negative b/c we're minimizing
    score = res[0.6][0]['micro']
    #print 'micro-f1: %.3f' % score
    return -1.*score


class DwOpt:
    """
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """
    def __init__(self, input_dim, bounds=None, default_params=None, sd=None, run_id=None):
        self.input_dim = input_dim
        self.bounds = bounds
        self.default_params = default_params
        self.evals, self.fails = 0, 0
        if sd == None: 
            self.sd = 0
        else: 
            self.sd = sd
        if run_id:
            run_dir = join(base_dir, 'bo', run_id)
            if not exists(run_dir):
                os.makedirs(run_dir)
            self.fout = open(join(run_dir, 'log.json'), 'w+')
        else:
            self.fout = None
        self.params = []

        # names of optimized parameters
        self.opt_names = [x['name'] for x in self.bounds]
        #self.other_params = dict((k,v) for k,v in default_params.items() if k not in self.output_file)

    def vec2dict(self, X):
        # add parameters being optimized
        d = {}
        for i, dom in enumerate(self.bounds):
            k = dom['name']
            typ = self.default_params[k][1]
            d[k] = typ(X[i])

        # add defaults for any parameters not being optimized
        for k, (val, typ) in self.default_params.items():
            if k not in d:
                d[k] = typ(val)
        return d

    def f(self, X):
        X = np.reshape(X, self.input_dim)
        n = X.shape[0]
        assert n == len(mixed_domain)
        params = self.vec2dict(X)

        self.evals += 1
        params['eval'] = self.evals
        try:
            score = run_from_params(params)
            params['score'] = score
        except:
            # these will be missing rows in the log dataframe
            self.fails += 1
            print('\nFAILURE IN OBJECTIVE FUNCTION')
            return 0.

        if self.fout is not None:
            self.fout.write(json.dumps(params) + '\n')
            self.params.append(params)
        
        # print the param values and objective value
        df = pd.DataFrame([params]).loc[:, ['eval'] + self.opt_names + ['score']]
        if self.evals == 1:
            print df.to_string(index=False)
        else:
            print df.to_string(index=False).split('\n')[-1]

        return score

    def shutdown(self):
        if self.fout is not None:
            self.fout.close()


def save_opt(opt, run_id):
    outdir = join(base_dir, 'bo', run_id)
    opt.save_models(join(outdir, 'models.txt'))
    opt.save_evaluations(join(outdir, 'evals.txt'))
    opt.save_report(join(outdir, 'report.txt'))

    #print opt.X
    print '\nOptimal solution:'
    print opt.x_opt
    #print '\nObjective function at solution:'
    #print opt.f(opt.x_opt)


def read_log(run_id):
    fname = join(base_dir, join(base_dir, 'bo', run_id, 'log.json'))
    rows = []
    with open(fname) as f:
        for l in f:
            rows.append(json.loads(l.rstrip()))
    return pd.DataFrame(rows)


def read_evals(run_id):
    fname = join(base_dir, join(base_dir, 'bo', run_id, 'evals.txt'))
    df = pd.read_csv(fname, sep='\t').sort_values(by='Y')
    return df


if __name__=='__main__':
    if len(sys.argv) >= 2:
        action = sys.argv[1]
    else:
        action = ''

    print('='*100)
    print('action: %s' % action)

    if action == 'example':
        "Example from docs on multiple types of parameters."
        import GPyOpt

        run_id = 'example'

        func = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=5) 
        
        mixed_domain =[
           {'name': 'var1_2', 'type': 'continuous', 'domain': (-10,10),'dimensionality': 2},
           {'name': 'var3', 'type': 'continuous', 'domain': (-8,3)},
           {'name': 'var4', 'type': 'discrete', 'domain': (-2,0,2)},
           {'name': 'var5', 'type': 'discrete', 'domain': (-1,5)}]

        myBopt = GPyOpt.methods.BayesianOptimization(f=func.f,
                                                     domain=mixed_domain,
                                                     initial_design_numdata=20,
                                                     acquisition_type='EI',
                                                     exact_feval = True)

        max_iter = 10
        max_time = 60
        myBopt.run_optimization(max_iter, max_time)
        save_opt(myBopt, run_id)

    if action == 'dw':
        "Run for DwOpt job defined above."

        run_id = 'dw3'

        print('run_id: %s' % run_id)
        print('constraints:')
        print(mixed_domain)

        G, labels_matrix = load_blogcat()

        input_dim = len(mixed_domain)
        func = DwOpt(input_dim, bounds=mixed_domain, default_params=default_params, run_id=run_id)

        bo = BayesianOptimization(f=func.f,                  # function to optimize       
                                 domain=mixed_domain,        # box-constrains of the problem
                                 initial_design_numdata=2,  # number data initial design
                                 acquisition_type='EI',      # Expected Improvement
                                 exact_feval=True,           # True evaluations00
                                 verbosity=True)

        max_iter = 1
        max_time = None
        bo.run_optimization(max_iter, max_time)
        save_opt(bo, run_id)
        
        func.shutdown()
        print('Number of fails', func.fails)


    if action == 'load':
        #df1 = read_log('dw2')
        #print df1
        if len(sys.argv) >= 3:
            run_id = sys.argv[2]
        else:
            raise Exception('Must specify a second run_id argument.')
        df = read_evals(run_id)
        print df

    print('Done')
    print('='*100)

