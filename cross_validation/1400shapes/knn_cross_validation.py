#!/usr/bin/python3

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import neighbors,decomposition,cross_validation,pipeline,preprocessing
import sys
import pickle
from os import urandom

n_pca = int(sys.argv[2])
infile = sys.argv[1]
db = pickle.load(open(infile,"rb"))
# nome das figuras
data = np.array([db[i] for i in db.keys()])

Y = data[:,0].astype(int)
X = preprocessing.scale(data[:,1:])
clf = pipeline.Pipeline([('pca',decomposition.PCA(n_components = n_pca,whiten = True)),('knn',neighbors.KNeighborsClassifier(n_neighbors = 3))])

#it = cross_validation.KFold(Y.size,n_folds = 500)

#it = cross_validation.StratifiedShuffleSplit(Y,10,test_size = 0.5)

scores = ["precision_weighted","recall_weighted","f1_weighted"]

for s in scores:
 sc = []   
# print(s)
 for i in range(50):
  rand = int.from_bytes(urandom(4),'big')
  it = cross_validation.StratifiedKFold(Y,n_folds = 5,shuffle = True,random_state = rand)  
  l = cross_validation.cross_val_score(clf,X,Y,cv = it,scoring = s)
  sc = sc + list(l)
#  print("{0}: {1} {2}".format(i,l.mean(),l.std()))
 print("{0} total: {1} {2}".format(s,np.mean(sc),np.std(sc)))
