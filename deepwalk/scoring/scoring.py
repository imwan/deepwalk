"""
Script that demonstrates the multi-label classification used for BlogCatalog.
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

    labels = np.argwhere(labels_matrix)
    label_cnts = pd.Series(labels[:,1]).value_counts()

    if verbose > 1:
        print('\nLabel counts:')
        print(label_cnts)

    # delete the least frequent labels, which causes balancing problems
    labels_matrix = labels_matrix[:, :-2]

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
    number_shuffles = 1
    for x in range(number_shuffles):
        # if we just have one group, make the split the same every time
        if number_shuffles == 1:
            shuffles.append(skshuffle(features_matrix, labels_matrix, random_state=123))
        else:
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


if __name__=='__main__':

    G, labels_matrix = load_blogcat()

    if 1:
        "Test BlogCatalog embeddings as originally generated by DeepWalk CLI."

        import os
        embeddings_file = "../../example_graphs/blogcatalog.vec"

        if not exists(embeddings_file):
            os.system("python ../__main__.py --input ../../example_graphs/blogcatalog.mat " 
                      " --format mat --output %s" % embeddings_file)

        res = eval_blogcat(embeddings_file, G=G, 
                       labels_matrix=labels_matrix, 
                       training_percents=[0.6],
                       normalize=1, verbose=3)

        """
        Train percent: 0.6
        OrderedDict([
        ('micro', 0.32951979527321995), 
        ('macro', 0.19558691525865063), 
        ('samples', 0.37904536271809003), 
        ('weighted', 0.30567303598406398)])
        """



        
