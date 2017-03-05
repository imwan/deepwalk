"""
Test Sklearn cross validation pipeline.
"""
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

#estimators = [('reduce_dim', PCA()), ('clf', SVC())]
#pipe = Pipeline(estimators)
pipe = make_pipeline(PCA(whiten=False), SVC())

## Parameters of the estimators in the pipeline can be accessed using the <estimator>__<parameter> syntax:
params = dict(pca__n_components=[2, 3], svc__C=[0.1, 10, 100])

clf = GridSearchCV(pipe, param_grid=params)
clf.fit(iris.data, iris.target)

import pandas as pd

df = pd.DataFrame(clf.cv_results_)
df2 = df.loc[:, ['mean_test_score', 'rank_test_score', 'params']].set_index('rank_test_score').sort_index()
print df2