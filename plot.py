from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)
means = [[2,2],[8,3],[3,6]]
cov = [[1,0],[0,1]]
N = 500
K = 3
#multivariate_normal
X0 = np.random.multivariate_normal(means[0],cov,N)
X1 = np.random.multivariate_normal(means[1],cov,N)
X2 = np.random.multivariate_normal(means[2],cov,N)
X = np.concatenate((X0,X1,X2),axis = 0)
origin_label = np.asarray([0]*N + [1]*N + [2]*N)
def display(X,label):
	X0 = X[label == 0,:]
	X1 = X[label == 1,:]
	X2 = X[label == 2,:]
	plt.plot(X0[:,0],X0[:,1],'b^',markersize = 4,alpha = .8)
	plt.plot(X1[:,0],X1[:,1],'go',markersize = 4,alpha = .8)
	plt.plot(X2[:,0],X2[:,1],'rs',markersize = 4,alpha = .8)
	plt.axis('equal')
	plt.show()
display(X,origin_label)
# Gán nhãn mới
def khoitaocenter(X,k):
	return X[np.random.choice(X.shape[0],k,replace = False)]
def gannhan(X,center):
	D = cdist(X,center)
	return np.argmin(D,axis = 1)
def tammoi(X,label,K):
	center = np.zeros((K,X.shape[1]))
	for k in range (K):
		Xk = X[label == k,:]
		center[k,:] = np.mean(Xk,axis = 0)
	return center
def check(center,new_center):
	return (set([tuple(a) for a in center]) == 
        set([tuple(a) for a in new_center]))
def k_means(X,K):
	center = [khoitaocenter(X,K)]
	label = []
	it = 0
	while True:
		# gan nhan
		label.append(gannhan(X,center[-1]))
		# cap nhap tam
		new_center = tammoi(X,label[-1],K)
		if check(center[-1],new_center):
			break
		it += 1
		center.append(new_center)
	return (center,label,it)

(center,label,it) = k_means(X,K)
print("3 tam tim dc la: ")
print(center[-1])
display(X,label[-1])