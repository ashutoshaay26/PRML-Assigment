from __future__ import division

from scipy import exp
from scipy.linalg import eigh
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(filename):
	data=pd.read_csv(filename,header=None)
	x=np.array(data)
	return x


#Calculate Covariance Matrix
def cal_covariance(a):
    '''
    Input : a is nxd numpy matrix
    Output : A covariance matrix of dimention dxd
    '''
    n,d = np.shape(a)
    mean_matrix = np.ones((n,d))*cal_mean(a)
    co_var = (1/n) * (np.transpose(a-mean_matrix).dot(a-mean_matrix))
    return co_var


#Calculate Mean
def cal_mean(a):
    '''
    Input : a is nxd numpy matrix
    Output : A mean vector of dimention d
    '''
    n,d = np.shape(a)
    #print(n,d)
    mean = np.sum(a,axis=0) / n
    return(mean)

#Calculate Log Likelihood
def cal_loglikelihood(x,mean,cov):
	n= x.shape[0]
	total_point_sum=0
	for i in range(n):
		total_point_sum += np.dot(np.dot((x[i]-mean),np.linalg.inv(cov)),(x[i]-mean).T)
	answer = (-1/2)*( n*2*(np.log(2*np.pi)) + np.log(np.linalg.det(cov)) + total_point_sum )
	return answer
def plot_data(data):
	ax = sns.scatterplot(x=data[:,0],y=data[:,1])
	plt.title("Scatter plot of Dataset-1")
	plt.xlabel("Dimention X")
	plt.ylabel("Dimention Y")
	plt.show()

def plot_vary_mean(data):
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	import numpy as np
	fig = plt.figure(figsize=(15,10))
	ax = fig.gca(projection='3d')

	# Make data.
	X = np.arange(-10, 10.5, 0.5)
	Y = np.arange(-10, 10.5, 0.5)
	XX, YY = np.meshgrid(X, Y)
	covariance = np.array([[4.638191501539513, 1.4543735240304285], [1.4543735240304285, 0.6557003296295485]])
	myi = -1000000000

	z_final = []
	a = 0
	b = 0
	for i in X:
		z_semi= []
		for j in Y:
			mean = np.array([i,j])
			z = cal_loglikelihood(data,mean,covariance)
			if(myi<z):
				a = i
				b =j
				myi = z
			z_semi.append(z)
		z_final.append(z_semi)
	# Plot the surface.
	surf = ax.plot_surface(np.array(XX), np.array(YY), np.array(z_final), cmap=cm.coolwarm,
	                     linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title("Log Likelihood ")
	plt.xlabel("x component of mean")
	plt.ylabel("y component of mean")

	plt.show()
	
if __name__ == "__main__":
	file_path = "PRML_assignment1/Datasets/Dataset1.csv"
	x=read_data(file_path) # Read the data. Return numpy array
	
	mean = np.mean(x, axis=0)
	cov = cal_covariance(x)
	#print(cal_loglikelihood(x,mean,cov))
	plot_vary_mean(x)
