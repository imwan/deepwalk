from os.path import join, exists
import random
import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import VectorizerMixin

from smm.wordvecs import run_gensim, run_fasttext

from deepwalk import graph
from scoring import load_blogcat


class DwWalker(BaseEstimator, TransformerMixin):
    def __init__(self, num_paths=10, path_length=40):
        self.num_paths = num_paths
        self.path_length = path_length

    def fit_transform(self, G, y=None, **fit_params):
        walk_params = self.get_params()
        print walk_params
        
        bd = '/mnt/raid1/nlp/deepwalk/walks/blogcatalog/'
        walks_file = join(bd, 'walks-%(num_paths)ix%(path_length)i-paths.txt' % \
                         walk_params)

        if False: #exists(walks_file):
            print('Walk path already exists: %s' % walks_file)
        else:
            print('Walking...')
            walks = graph.build_deepwalk_corpus(G, **walk_params)

            walks_flat = [node for path in walks for node in path]

            print('Writing {:,} walks...'.format(len(walks_flat)))
            with open(walks_file, 'w+') as f:
                f.write(' '.join(walks_flat))
        return walks_file


class DwVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05, size=100, window=5, iter=5, min_count=0, 
                 sample=1e-4, sg=1, hs=0, negative=5, loss='ns', 
                 corpus_file='/tmp/DwWalks.txt',
                 output_file='/tmp/DwEstimator.vecs',
                 normalize=1, verbose=1):

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
        self.corpus_file = corpus_file
        self.output_file = output_file
        self.normalize = normalize
        self.verbose = verbose

    def fit(self, X, y=None):

        dw_params = self.get_params()
        print dw_params

        if False: #exists(self.output_file):
            model = KeyedVectors.load_word2vec_format(self.output_file)
        else:
            model = run_gensim(dw_params)
        nb_vecs = len(model.wv.vocab)

        # Map nodes to their features (note: assumes nodes are labeled as integers 1:N) 
        features_matrix = np.asarray([model[str(node)] for node in range(nb_vecs)])
        #features_matrix = np.random.randn((4,2))

        if self.normalize:
            norms = np.linalg.norm(features_matrix, axis=1)
            if self.verbose:
                print norms
                print norms.shape

            assert norms.shape[0] == features_matrix.shape[0]
            for i in range(features_matrix.shape[0]):
                features_matrix[i,:] /= norms[i]

            norms = np.linalg.norm(features_matrix, axis=1)
            if self.verbose:
                print norms

        if self.verbose:
            print('features_matrix.shape = %s' % str(features_matrix.shape))

        self.dw_params_ = dw_params
        self.gs_model_ = model
        self.features_matrix_ = features_matrix
        print('fit', self.features_matrix_.shape)
        return self

    def transform(self, X, y=None):
        print('transform', self.features_matrix_.shape)
        return self.features_matrix_

    def fit_transform(self, X, y=None, **fit_params):
        print('-'*100)
        print('fit_params:')
        print(fit_params)
        res = self.fit(X).transform(X)
        print('fit_transform', res.shape)
        return res
         

# class TopKRanker(OneVsRestClassifier):
#     def predict(self, X, top_k_list):
#         assert X.shape[0] == len(top_k_list)
#         probs = np.asarray(super(TopKRanker, self).predict_proba(X))
#         all_labels = []
#         for i, k in enumerate(top_k_list):
#             probs_ = probs[i, :]
#             labels = self.classes_[probs_.argsort()[-k:]].tolist()
#             all_labels.append(labels)
#         # convert list of lists to sparse encoding
#         return MultiLabelBinarizer().fit_transform(all_labels)

#     def score(self, X, y, sample_weight=None):

class JLR(LogisticRegression):
    def fit(self, X, y=None):
        print('-'*100)
        print('X', X.shape, 'y', y.shape)
        super(JLR, self).fit(X, y=y)


if __name__ == '__main__':    
    # pipes = [('walk', DwWalker()), 
    #          ('vec ', DwVectorizer()),
    #          ('clf', DwClassifier())]
    # pipeline = Pipeline(pipes)
    # params = dict(walk__num_paths=[10, 20])

    G, labels_matrix = load_blogcat()

    walker = DwWalker()
    walks_file = walker.fit_transform(G)

    # vec = DwVectorizer(corpus_file=walks_file, iter=1, verbose=0)
    # X = vec.fit_transform(None)

    # delete the least frequent labels, which causes balancing problems
    #Y = labels_matrix[:, :-5]
    Y = labels_matrix[:, 7]
    Y = np.array(Y.flatten())[0] # flatten into single vector

    # fake some X data
    #X = np.zeros((Y.shape[0], 100)) 
    X = np.random.randn(Y.shape[0], 100)
    
    print('X', str(X.shape))
    print('Y', str(Y.shape))

    pipes = [('vec', DwVectorizer(corpus_file=walks_file, iter=1, verbose=1)),
              ('clf', JLR())]
    pipeline = Pipeline(pipes)

    params = dict(vec__window=[5,6])
    clf = GridSearchCV(pipeline, param_grid=params, verbose=100, 
                       cv=StratifiedKFold(n_splits=2))
    clf.fit(X, Y)

    # df = pd.DataFrame(clf.cv_results_)
    # df2 = df.loc[:, ['mean_test_score', 'rank_test_score', 'params']].set_index('rank_test_score').sort_index()
    # print df2




