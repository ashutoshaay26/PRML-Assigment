# PRML Assignment 3
# Question 23
# Gradient Descent for Ridge Regression 

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

# Split the data into train and test data.
def split_data(x,y,ratio=0.2):
	s = int(x.shape[0]*0.2)
	train_data =  x[:s,:]
	train_label = y[:s]
	val_data =  x[s+1:,:]
	val_label = y[s+1:]

	return train_data,train_label,val_data,val_label

# Closed Form Solution w = (x.T*x)^-1 * x.T * Y 
def closed_form_gd(x,y):
	return np.dot(np.linalg.inv(np.dot(np.transpose(x),x)) , np.dot(np.transpose(x),y))


# Gradient Descent for ridge regression
def gradient_descent(x,y,weights,lamb,step_size,epoch=100):

	#weights_t =[weights]
	#cost=[]
	for i in range(epoch):
		pred = np.dot(data,weights)

		#calculating Error
		error = pred - label

		# Update the weights (in matrix form)
		weights = weights - step_size * ((1/data.shape[0]) * np.dot(np.transpose(data),error)+ lamb*weights )
		#weights_t.append(weights)
		#cost.append(error)
	return weights

# Return the list of error for different set of lambdas.
def cross_validation_lamb(train_data,train_label,val_data,val_label,w,list_lambda,step_size,epoch=100):
	error_list=[]
	current_cost=1000000
	best_lambda=0.1
	best_weights=w
	for i in range(len(list_lambda)):
		w = gradient_descent(train_data,train_label,w,list_lambda[i],step_size,epoch)
		temp_error = compute_error_r(val_data,val_label,w,list_lambda[i])
		error_list.append(temp_error)
		if temp_error < current_cost:
			best_lambda=list_lambda[i]
			best_weights=w
	return error_list,best_lambda,best_weights

# Computer error using loss function for closed form solution w_ml
def compute_error_ml(x,y,w):
	cost = (1/(2*x.shape[0])) * (np.sum( (np.dot(x,w) - y)**2 )  )
	return cost

# Computer error using loss function for Ridge rigression solution w_r
def compute_error_r(x,y,w,lamb):
	cost = (1/(2*x.shape[0])) * (np.sum( (np.dot(x,w) - y)**2 ) + lamb * np.dot(np.transpose(w),w) )
	return cost



# Plot Test Error.
def plot_test_error(test_x,test_y,w_r,w_ml,best_lambda):
	error_r = compute_error_r(test_x,test_y,w_ml,best_lambda)
	error_ml = compute_error_ml(test_x,test_y,w_ml)
	plt.plot(error_r,'r')
	plt.label("Test Error W_r")
	plt.plot(error_ml,'b')
	plt.label("Test Error W_ml")

	plt.title("Test Error")
	plt.xlabel("Timestem")
	plt.ylabel("Prediction Cost")
	plt.show()


# Plot Cross Validation error for different lambdas.
def plot_cross_val_lambda(error,list_lambda,step_size,epoch):
	plt.style.use('seaborn-whitegrid')
	
	plt.plot(list_lambda,error)
	plt.title("Cross Validation for different valus of lambdas "+"step size:"+str(step_size) +" Epoch:"+str(epoch))
	plt.xlabel("Lambda value")
	plt.ylabel("Validation Error")
	plt.show()


if __name__ == "__main__":
	file_path1 = "Data/Dataset_train.csv"
	data,label=read_data(file_path1) # Read the data. Return numpy array

	file_path2 = "Data/Dataset_test.csv"
	test_data,test_label=read_data(file_path2) # Read the data. Return numpy array

	
	# Parameter Initialization	
	epoch=100
	step_size=0.01
	batch_size=100
	split_ratio=0.2
	list_lambda = [0.01,0.07,0.2,0.5,0.9,1,1.5,1.8,2]

	weights = np.random.rand(100)
	
	train_data,train_label,val_data,val_label = split_data(data,label,split_ratio)

	
	error_list,best_lambda,w_r = cross_validation_lamb(train_data,train_label,val_data,val_label,weights,list_lambda,step_size,epoch)


	plot_cross_val_lambda(error_list,list_lambda,step_size,epoch)

	w_ml=closed_form_gd(train_data,train_label)


	print("Test Error, w_ml : ",compute_error_ml(test_data,test_label,w_ml))
	print("Test Error, w_r : ",compute_error_r(test_data,test_label,w_r,best_lambda))