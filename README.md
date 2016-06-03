import mdtraj as md #For uploading Tajectories
import msmbuilder as msm #for builing msm
import numpy as np #for handling aarays
traj = md.load('r9par157.mdcrd', top = 'stripped.prmtop') #uploading Trajecories and topology file
from msmbuilder.featurizer import DihedralFeaturizer #For Dihedral Featurization
feat = DihedralFeaturizer(types = ['phi', 'psi'], sincos = True) #Defining what we ant to calc phi, psi, sincos 
seq = feat.transform(list(traj)) #will give the sincos values of the each of the frame of each trajectory
from msmbuilder.decomposition import tICA
obj1 = tICA(n_components = 10,lag_time=50,weighted_transform=False)
seq2=[i[0] for i in seq]
seq3=[seq2]
seq3[0]=np.array(seq3[0])
tics=obj1.fit(seq3)
tics.components_
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=5).fit(tics.components_)
l=cluster.labels_
from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump
model = MarkovStateModel(lag_time=20)
model.fit(l)

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
