# PRML Assignment 2
# Question 2 
# Gradient Descent 

'''
-- Conventions : 
	data matrix :: nxd
	weight matrix :: dx1
	label matrix :: nx1

'''

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

def read_data(filename):
	x = np.array(pd.read_csv(filename,header=None))
	data = x[:,:-1]
	label= x[:,-1]
	del x

	# data : nxd, label : nx1
	return (data,label)


#Gradient Descent Algorithm.
def gradient_descent(data,label,weights,step_size,epoch=100):

	weights_t =[weights]
	#cost=[]
	for i in range(epoch):
		pred = np.dot(data,weights)

		#calculating Error
		error = pred - label

		# Update the weights (in matrix form)
		weights = weights - step_size * (1/data.shape[0]) * np.dot(np.transpose(data),error)
		weights_t.append(weights)
		#cost.append(error)
	return weights_t

#Stochastic Gradient Descent Algorithm.
def stochastic_gradient_descent(data,label,weights,step_size,batch_size=100,epoch=100):
	weights_t=[weights]
	#cost=[]
	for i in range(epoch):
		#temp_cost=0
		for j in range(0,data.shape[0],batch_size):
			pred = np.dot(data[j:j+batch_size,:],weights)
			#calculating Error
			error = pred - label[j:j+batch_size]
			#temp_cost+=error
			# Update the weights (in matrix form)
			weights = weights - step_size * (1/batch_size) * np.dot(np.transpose(data[j:j+batch_size,:]),error)
			weights_t.append(weights)
		#cost.append(temp_cost/(data.shape[0]//batch_size))		
	return weights_t


# Closed Form Solution w = (x.T*x)^-1 * x.T * Y 
def closed_form_gd(x,y):
	return np.dot(np.linalg.inv(np.dot(np.transpose(x),x)) , np.dot(np.transpose(x),y))

# plot data
def plot_data_two(w_t,w_ml):
	cost = []
	for i in w_t:
		cost.append(np.linalg.norm(i-w_ml))

	plt.style.use('seaborn-whitegrid')
	ax = plt.plot(cost)
	plt.title("Absolute difference ||w_t - w_ml||, ")
	plt.xlabel("Timestemp")
	plt.ylabel("Frobenious Norm")
	plt.show()
	#plt.savefig("")
	#plt.close()

def plot_cost(c1,c2):

	#plt.style.use('seaborn-whitegrid')
	plt.plot(c1,'r')
	#plt.plot(c2,'g')
	plt.title("Test Error")
	plt.xlabel("Timestemp")
	plt.ylabel("Prediction Cost")
	plt.show()


if __name__ == "__main__":
	file_path = "Data/Dataset_train.csv"
	data,label=read_data(file_path) # Read the data. Return numpy array
	
	epoch=100
	step_size=0.03
	batch_size=100
	weights = np.random.rand(100)
	
	#w_t = gradient_descent(data,label,weights,step_size,epoch)
	w_t = stochastic_gradient_descent(data,label,weights,step_size,batch_size,epoch)
	w_ml=closed_form_gd(data,label)
	plot_data_two(w_t,w_ml)
	#plot_cost(cost1,cost2)