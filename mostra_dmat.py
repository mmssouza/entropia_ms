#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:56:00 2019

@author: msouza
"""

import pylab
import pickle
import sys
from scipy.spatial.distance import pdist,squareform
from sklearn import decomposition,preprocessing

db = pickle.load(open(sys.argv[1],"rb"))
data = pylab.array([v for v in db.values()])
X = pylab.vstack([data[pylab.where(data[:,0] == i)] for i in range(1,int(data[:,0].max())+1)])
print(X.shape)
X = preprocessing.scale(X[:,1:])
print(X.shape)
XX = decomposition.PCA(n_components = 4,whiten = True).fit_transform(X)
print(XX.shape)
dist = squareform(pdist(XX))
pylab.imshow(dist,cmap=pylab.get_cmap('gist_gray'))
pylab.show()
