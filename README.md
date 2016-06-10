%cpaste
import mdtraj as md
import msmbuilder as msm
import numpy as np
import glob
from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump
from msmbuilder.decomposition import tICA
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.cluster import KMeans
feat = DihedralFeaturizer(types = ['phi', 'psi'], sincos = True)
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")
import random
import math
import operator
#all libraries have been imported now
traj=[]
for file in glob.glob('*1.dcd*'):
     temp=md.load(file, top = 'pnas2011b-A-00-all.pdb')
     traj.append(temp)
seq = feat.fit_transform(traj)
obj1 = tICA(n_components = 10,lag_time=50,weighted_transform=False)
tics=obj1.fit_transform(seq)
cluster = KMeans(n_clusters=50).fit(tics)
l=cluster.labels_
model = MarkovStateModel(lag_time=20)
model.fit(l)
n = model.n_states_
q = model.mapping_
a = list(q)
a.sort()
totaldiahedrals = len (seq[0][0])
appolo = []
clf = svm.SVC(kernel = 'linear',C = 1000000.0)
firsteigenvector = model.left_eigenvectors_[:,1]

def dist(m,n,a,b):
	distan = abs(a*m-n+b)
	return distan
Collection = []
finalset = []
for j in range(0,n):
	p=[]
	for i in range(0,len(traj)):
			p.append([i,np.where(l[i] == a[j])[0]])
	indx = []
	for i in range(0,len(traj)):
		if len(p[i][1]) != 0 :
			indx.append(i)
	r = [] 
	for i in range(0,len(indx)):
		r.append(np.column_stack((np.array([indx[i]]*len(p[indx[i]][1])),p[indx[i]][1])).tolist())
	Collection.append([r])
	finalset.append(random.choice(r))

Y = []
for i in range (0,n):
	if firsteigenvector[i] >= 0:
		Y.append(1)
	else:
		Y.append(0)

#The code above this is getting the frames corresponding to a particular state.

for ticno in range(0,totaldiahedrals):
	t2ofall = []
	for i in range(0,n):
		t2ofall.append(seq[finalset[i][0]][finalset[i][1]][ticno])
	X = np.column_stack((t2ofall, firsteigenvector))
	
	clf.fit(X,Y)
	w = clf.coef_[0] 
	A = -w[0]/w[1] 
	B = -clf.intercept_[0]/w[1]

	yelo = 0
	for i in range(0,n):
		yelo = yelo + dist(X[i,0], X[i, 1],A,B)
	yelo = yelo/ math.sqrt(math.pow(A, 2)+1)
	appolo.append(yelo)

mostimpdia = appolo.index(max(appolo))

t2ofall = []
for i in range(0,n):
	t2ofall.append(seq[finalset[i][0]][finalset[i][1]][mostimpdia])
X = np.column_stack((t2ofall, firsteigenvector))
clf.fit(X,Y)
w = clf.coef_[0] 
A = -w[0]/w[1] 
B = -clf.intercept_[0]/w[1]
xx = np.linspace(X.min(),X.max()) 
yy = A*xx + B 
h0 = plt.plot(xx, yy, 'k-', label = n)  
plt.scatter(X[:,0], X[:, 1], c = Y) 
plt.legend() 
plt.savefig('SVMimplement') 
number = np.array(list(range(0,totaldiahedrals)))
appoloarray = np.array(appolo)
indexing = np.column_stack((number, appoloarray))

forsortingappolo = appolo[:]
forsortingappolo.sort()
indexing= []
for i in range(0,totaldiahedrals):
	m = forsortingappolo[i]
	indexing.append(appolo.index(m))
indexing.reverse()
--
