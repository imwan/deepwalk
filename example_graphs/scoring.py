"""
Script that demonstrates the multi-label classification used.
"""
from os.path import join, exists
import random
import numpy as np
import pandas as pd
from itertools import izip
from collections import defaultdict, OrderedDict

from scipy.io import loadmat

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

from gensim.models import Word2Vec, KeyedVectors

from deepwalk import graph


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        # convert list of lists to sparse encoding
        return MultiLabelBinarizer().fit_transform(all_labels)


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.iteritems()}


def load_blogcat():
    "Load BlogCatalog labels from mat file."
    matfile = "/home/jimmie/git/deepwalk/example_graphs/blogcatalog.mat"
    print('\nLoading BlogCatalog from mat file: %s' % matfile)
    G = graph.load_matfile(matfile, variable_name='network', undirected=1)
    G = G.stringify()

    mat = loadmat(matfile)
    #A = mat['network']
    #graph = sparse2graph(A)
    
    labels_matrix = mat['group']
    labels_matrix = labels_matrix.todense().astype(np.int32)
    return G, labels_matrix   


def eval_blogcat(embeddings_file, labels_matrix=None, G=None,
                 verbose=1, normalize=1, training_percents=[0.1, 0.6, 0.9]):

    # 0. Files
    #embeddings_file = "/mnt/raid1/deepwalk/blogcatalog.vec"
    if labels_matrix is None and G is None:
        G, labels_matrix = load_blogcat()
    
    # 1. Load Embeddings
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

    if verbose > 1:
        print('\nLabel counts:')
        labels = np.argwhere(labels_matrix)
        print pd.Series(labels[:,1]).value_counts()

    # delete the least frequent labels, which causes balancing problems
    labels_matrix = labels_matrix[:, :-5]

    # Map nodes to their features (note: assumes nodes are labeled as integers 1:N) 
    features_matrix = np.asarray([model[str(node)] for node in range(len(G))])

    if normalize:
        norms = np.linalg.norm(features_matrix, axis=1)
        if verbose:
            print norms
            print norms.shape

        assert norms.shape[0] == features_matrix.shape[0]
        for i in range(features_matrix.shape[0]):
            features_matrix[i,:] /= norms[i]

        norms = np.linalg.norm(features_matrix, axis=1)
        if verbose:
            print norms

    if verbose:
        print('-'*100)
        print(embeddings_file)
        print('features_matrix.shape = %s' % str(features_matrix.shape))
        print('labels_matrix.shape   = %s' % str(labels_matrix.shape))

    # 2. Shuffle, to create train/test groups
    shuffles = []
    number_shuffles = 1 # max tries before quitting
    for x in range(number_shuffles):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    # uncomment for all training percents
    #training_percents = np.asarray(range(1,10))*.1
    for train_percent in training_percents:
        # print('-'*100)
        # print('pct_train: %.2f' % train_percent)

        for shuf in shuffles:
            X, y = shuf
            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train = y[:training_size]
            X_test = X[training_size:, :]
            y_test = y[training_size:]

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train)

            # find out how many labels should be predicted
            #top_k_list = [len(l) for l in y_test]
            top_k_list = np.array(np.sum(y_test, axis=1).flatten()[0])[0].astype(np.int32)
            preds = clf.predict(X_test, top_k_list)

            if y_test.shape[1] != preds.shape[1]:
                raise Exception("imbalance of class dims")
                #continue
            
            results = OrderedDict()
            averages = ["micro", "macro", "samples", "weighted"]
            for average in averages:
                results[average] = f1_score(y_test, preds, average=average)

            all_results[train_percent].append(results)
            #break

    if verbose:
        print '-------------------'
        for train_percent in sorted(all_results.keys()):
            print 'Train percent:', train_percent
            for x in all_results[train_percent]:
                print  x
            print '-------------------'
    return all_results


def test_blogcat():
    embeddings_file = "/mnt/raid1/deepwalk/blogcatalog.vec"
    res = eval_blogcat(embeddings_file, verbose=1, normalize=1)
    return res


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV

from smm.wordvecs import run_gensim, run_fasttext


class DwEstimator(BaseEstimator, ClassifierMixin):

    # Parameters in the parlance of gensim's Word2Vec
    default_params = {
        'alpha': 0.05,
        'size': 100,
        'window': 5,
        'iter': 5,
        'min_count': 0,
        'sample': 1e-4,
        'sg': 1,
        'hs': 0,
        'negative': 5,
        'loss': 'ns',
        'output_file': '/tmp/DwEstimator.vecs',
        'num_paths': 10,
        'path_length': 40,
    }

    # list of parameters in the above that correspond to random walking
    walk_params = set(['num_paths', 'path_length'])

    def __init__(self, alpha=0.05, size=100, window=5, iter=5, min_count=0, sample=1e-4, 
                 sg=1, hs=0, negative=5, loss='ns', output_file='/tmp/DwEstimator.vecs',
                 num_paths=10, path_length=40):
                
        # print 'Input kwargs:'
        # print kwargs
        # for k in self.walk_params:
        #     if k not in kwargs:
        #         print 'missing random walk parameter %s' % k
        # for k,v in kwargs.items():
        #     setattr(self, k, v)
        self.alpha = alpha
        self.size = size
        self.window = window
        self.iter = iter
        self.min_count = min_count
        self.sample = sample
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.loss = loss
        self.output_file = output_file
        self.num_paths = num_paths
        self.path_length = path_length

        self.G_ = None
        self.labels_matrix_ = None

    def fit(self, X, y, **fit_params):
        "Fit the word vector model and write vectors to params['output_file']."

        assert isinstance(fit_params, dict), 'fit_params is not a dict'
        for k,v in fit_params.items():
            assert k in set(['G', 'labels_matrix'])
            setattr(self, k+'_', v)

        #walk_params = {'num_paths': self.num_paths, 'path_length': self.path_length}
        walk_params = dict((k, getattr(self, k)) for k in self.walk_params)
        walk_path = join('/mnt/raid1/nlp/deepwalk/walks/blogcatalog/walks-%(num_paths)ix%(path_length)i-paths.txt' % \
                         walk_params)

        if exists(walk_path):
            print('Walk path already exists: %s' % walk_path)
        else:
            print('Walking...')
            walks = graph.build_deepwalk_corpus(self.G_, **walk_params)

            print('Writing walks...')
            with open(walk_path, 'w+') as f:
                f.write(' '.join([node for path in walks for node in path]))

        print self.get_params()

        #dw_params = dict((k, getattr(self, k)) for k in self.default_params.keys())
        dw_params = self.get_params()
        dw_params['corpus_file'] = walk_path

        self.gs_model_ = run_gensim(dw_params)
        self.walk_path_ = walk_path
        # required by sklearn
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def score(self, X, y, sample_weight=None):
            
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        if not self.output_file.endswith('.vec'):
            self.output_file += '.vec'

        res = eval_blogcat(self.output_file, 
                           G=self.G_, 
                           labels_matrix=self.labels_matrix_, 
                           training_percents=[0.6],
                           normalize=1, verbose=0)

        s = res[0.6][0]['macro']
        
        print 'score = %.4f' % s
        return s

#check_estimator(DwEstimator)

if __name__=='__main__':


    G, labels_matrix = load_blogcat()

    if 1:
        "Test BlogCatalog embeddings as originally generated by DeepWalk CLI."
        embeddings_file = "/mnt/raid1/deepwalk/blogcat/models/model-base.vec"
        res = eval_blogcat(embeddings_file, G=G, 
                       labels_matrix=labels_matrix, 
                       training_percents=[0.6],
                       normalize=1, verbose=1)

    if 0:
        print '='*100

        fit_params = {}
        fit_params['G'] = G
        fit_params['labels_matrix'] = labels_matrix

        X, y = np.zeros((2,1)), np.zeros(2)

        est = DwEstimator(**DwEstimator.default_params)

        print('Fitting...')
        est.fit(X, y, **fit_params)

        print('Scoring...')
        s = est.score(None)

    if 0:
        print '='*100
        # param_grid = [
        #    {'num_paths': [5, 10, 20], 'path_length': [10, 40, 70],
        #     'alpha': [0.01, 0.05, 0.10], 'size': [32, 64, 100], 'window': [2, 5, 10],
        #     'iter': [2, 5, 10], 'negative': [2, 5, 10]
        #    }
        # ]
        param_grid = [
           {'num_paths': [10, 20]}
        ]

        gs_est = DwEstimator(**DwEstimator.default_params)
        gs = GridSearchCV(gs_est, param_grid, fit_params=fit_params)
        gs.fit(X, y)
        
