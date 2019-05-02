#!/usr/bin/python

print __doc__


# Code source: Gael Varoqueux
# Modified for Documentation merge by Jaques Grobler
# License: BSD

import numpy as np
from sklearn import neighbors,decomposition,cross_validation,pipeline,metrics
import sys
import cPickle
import warnings

warnings.simplefilter("ignore")

# import some data to play with

args = sys.argv[2]
db = cPickle.load(open(sys.argv[2]))

# nome das figuras
data = np.array([db[i] for i in db.keys()])

Y = data[:,0].astype(int)
X = data[:,1:]
clf = pipeline.Pipeline([('pca',decomposition.PCA(n_components = int(sys.argv[1]),whiten = True)),('knn',neighbors.KNeighborsClassifier(n_neighbors = 2))])

it = cross_validation.KFold(Y.size,n_folds = 4)

#it = cross_validation.StratifiedShuffleSplit(Y,150,test_size = 0.5)
#it = cross_validation.StratifiedKFold(Y,n_folds = 5)

l = cross_validation.cross_val_score(clf, X,Y,cv = it,scoring = "precision")
print  l.mean(),l.std()
l = cross_validation.cross_val_score(clf, X,Y,cv = it,scoring = "recall")
print  l.mean(),l.std()
l = cross_validation.cross_val_score(clf, X,Y,cv = it,scoring = "f1")
print  l.mean(),l.std()

y_pred = [] 
y_true = []
for a,b in it:
 clf.fit(X[a],Y[a])
 y_pred.append(clf.predict(X[b]))
 y_true.append(Y[b])

C = metrics.confusion_matrix(np.hstack(y_true),np.hstack(y_pred))
print C
  
