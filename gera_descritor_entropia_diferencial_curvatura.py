#! /usr/bin/python3

import sys
import pickle
from scipy.stats.kde import gaussian_kde
from scipy.integrate import quad
import math
import scipy
import scipy.stats
import descritores
import time


def desc_entropy(fn,s):
 k = descritores.curvatura(fn,s)

 #hist = scipy.vstack([scipy.stats.histogram(a,numbins = 2500,defaultlimits = (-250.,250.)) for a in k.curvs])
 h = []
 for c,contour in zip(k.curvs,k.contours):
#  print c.shape
#  print c.min(),c.max()
  #caux = scipy.tanh(c)
  caux = c
  my_pdf = gaussian_kde(caux)
  h.append(quad(lambda x: -my_pdf(x)*scipy.log(my_pdf(x)+1e-12),-3*caux.std(),3*caux.std())[0]+scipy.log(contour.perimeter()))
  #t = scipy.linspace(caux.min(),caux.max(),350)
#  h.append(simps(-my_pdf(t)*scipy.log(my_pdf(t)+1e-12),t) + scipy.log(contour.perimeter()))
 #H(my_pdf(x)/my_pdf(x).max())) 
 return scipy.array(h)

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
  db[im_file]  = scipy.hstack((cl[im_file],desc_entropy(diretorio+im_file,sigma)))

with open(sys.argv[2],'wb') as f:
 pickle.dump(db,f)

print("done feature extraction in {0} seconds".format(time.time()-tt))

 
