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

db = pickle.load(open(sys.argv[1],"rb"))
data = pylab.array([v for v in db.values()])
l = data[:,0]
data = pylab.vstack([data[pylab.where(l == i)] for i in l])
dist = squareform(pdist(data[:,1:]))
pylab.imshow(dist,cmap=pylab.get_cmap('gray'))
pylab.show()
