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
	pass


def plot_data(data):
	ax = sns.scatterplot(x=data[:,0],y=data[:,1])
	plt.title("Scatter plot of Dataset-1")
	plt.xlabel("Dimention X")
	plt.ylabel("Dimention Y")
	plt.show()


if __name__ == "__main__":
	file_path = "PRML_assignment1/Datasets/Dataset1.csv"
	x=read_data(file_path) # Read the data. Return numpy array
	
	mean = np.mean(x, axis=0)
	cov = np.cov(x, rowvar=0)
	print(mean)
	print(cov)
	print(cal_mean(x))
	print(cal_covariance(x))
	plot_data(x)
