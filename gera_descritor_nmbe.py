#! /usr/bin/python3

import sys
import pickle
import math
import scipy
import scipy.stats
import descritores
import time



sys.path.append("./")

diretorio = sys.argv[1]
f = open(diretorio+"classes.txt","rb")
cl = pickle.load(f)
f.close()

db = {}

# Definição as escalas de acordo com Costa (1996)
S = 16
tau_max = 128.
tau_min = 0.8

oct_l = scipy.array([math.sqrt(2)**l for l in range(S)])

sigma= tau_min + (tau_max - tau_min)*(oct_l - math.sqrt(2))/(oct_l.max() - math.sqrt(2))

print(sigma)
#sigma= scipy.logspace(0.1,1.65,N)
tt = time.time()

print("feature extraction")

for im_file in cl.keys():
  print(im_file)  
  nmbe = descritores.bendenergy(diretorio+im_file,sigma)  
  db[im_file]  = scipy.hstack((cl[im_file],scipy.log(nmbe())))

with open(sys.argv[2],'wb') as f:
 pickle.dump(db,f)

print("done feature extraction in {0} seconds".format(time.time()-tt))

 
