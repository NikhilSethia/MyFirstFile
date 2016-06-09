import mdtraj as md #For uploading Tajectories
import msmbuilder as msm #for builing msm
import numpy as np #for handling aarays
traj = md.load('r9par157.mdcrd', top = 'stripped.prmtop') #uploading Trajecories and topology file
from msmbuilder.featurizer import DihedralFeaturizer #For Dihedral Featurization
feat = DihedralFeaturizer(types = ['phi', 'psi'], sincos = True) #Defining what we ant to calc phi, psi, sincos 
seq = feat.transform(list(traj)) #will give the sincos values of the each of the frame of each trajectory
from msmbuilder.decomposition import tICA #importing tICA to reduce Dimensionality
obj1 = tICA(n_components = 10,lag_time=50,weighted_transform=False) #defining tICA object
seq2=[i[0] for i in seq] # handling seq
seq3=[seq2] #handling seq
seq3[0]=np.array(seq3[0]) #changing it to array
tics=obj1.fit(seq3) #defining tICA object
tics.components_ #to see the components that present now
from sklearn.cluster import KMeans #
cluster = KMeans(n_clusters=5).fit(tics.components_) # clustering
l=cluster.labels_ #getting labels
from msmbuilder.msm import MarkovStateModel #building MarkovState
from msmbuilder.utils import dump #importing dump
model = MarkovStateModel(lag_time=20) #making lag time = 20
model.fit(l) #fitting the data

#SVM code :-
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")
x = [1,5,1.5,8,1,9]
y = [2,8,1.8,8,0.6,11]
plt.scatter(x,y)
X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
y = [0,1,0,1,0,1]
clf = svm.SVC(kernel = 'linear',C = 1.0)
clf.fit(X,y)
clf.predict([0.58,0.76])
clf.predict([10.58,10.76])
w = clf.coef_[0]
print(w)
a = -w[0]/w[1]
xx = np.linspace(0,12)
yy = a*xx-clf.intercept_[0]/w[1]
h0 = plt.plot(xx, yy, 'k-', label = "non weighted div")
plt.scatter(X[:,0], X[:, 1], c = y)
plt.legend()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%cpaste

traj=[]
for file in glob.glob('*1.dcd*'):
     temp=md.load(file, top = 'pnas2011b-A-00-all.pdb')
     traj.append(temp)
--

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
ticno = 2

finalset = []
%cpaste

for j in range(0,len(a)):
	

	p=[]
	for i in range(0,9):
			p.append([i,np.where(l[i] == a[j])[0]])
	indx = []
	for i in range(0,9):
		if len(p[i][1]) != 0 :
			indx.append(i)
	r = [] 
	for i in range(0,len(indx)):
		r = np.column_stack((np.array([indx[i]]*len(p[indx[i]] [1])),p[indx[i]][1]))
	finalset.append(random.choice(r))

t2ofall = []
for i in range(0,len(finalset)):
	t2ofall.append(tics[finalset[i][0]][finalset[i][1]][ticno])
--

plt.scatter(t2ofall,model.left_eigenvectors_[:,1])
plt.savefig('FIG1')


X = np.column_stack((t2ofall, model.left_eigenvectors_[:,1]))
Y = []
%cpaste
for i in range (0,n):
	if model.left_eigenvectors_[:,1][i] >= 0:
		Y.append(1)
	else:
		Y.append(0)
--

clf = svm.SVC(kernel = 'linear',C = 1000000.0) 
clf.fit(X,Y) 
w = clf.coef_[0] 
a = -w[0]/w[1] 
xx = np.linspace(X.min(),X.max()) 
yy = a*xx-clf.intercept_[0]/w[1] 
h0 = plt.plot(xx, yy, 'k-', label = "non weighted div") 
plt.scatter(X[:,0], X[:, 1], c = Y) 
plt.legend()
plt.savefig('SVMimplement')
