# Name : Shaikh Moin Dastagir 
# Roll no : 16CS30033

import numpy as np 
import math
import array 
import random as rd
from numpy import *
import matplotlib.pyplot as plt 


X_original_train=[]
Y_original_train=[]
X_original_test=[]
Y_original_test=[]



def Graph(x,y,n):
	# plotting the points  
	# plotting points as a scatter plot 
	plt.scatter(x, y, label= "stars", color= "green",marker= "*", s=30) 
	  
	# x-axis label 
	plt.xlabel('x - axis') 
	# frequency label 
	plt.ylabel('y - axis') 
	# plot title 
	plt.title('Plot for the generated dataset ' + ' for dataset of size ' + str(n)) 
	# showing legend 
	plt.legend() 
	  
	# function to show the plot 
	plt.show()


def Graph_2(x,y1,y2,n,m):
	# plotting the points  
	# plotting points as a scatter plot 
	plt.scatter(x, y1, label= "Real Values", color= "green",marker= "*", s=30) 
	plt.scatter(x, y2, label= "Predicted values", color= "red",marker= "*", s=30) 
	  
	# x-axis label 
	plt.xlabel('x - axis') 
	# frequency label 
	plt.ylabel('y - axis') 
	# plot title 
	plt.title('For n= '+ str(n) + ' for dataset of size ' + str(m)) 
	# showing legend 
	plt.legend() 
	  
	# function to show the plot 
	plt.show()


def Graph_3(x,y1,y2,n):
	# plotting the points  
	# plotting points as a scatter plot 
	plt.scatter(x, y1, label= "Train-error", color= "green",marker= "*", s=75)
	plt.scatter(x, y2, label= "Test-error", color= "red",marker= "*", s=75) 
	  
	# x-axis label 
	plt.xlabel('Data set Size') 
	# frequency label 
	plt.ylabel('Error') 
	# plot title 
	plt.title('Plot for the generated ' + ' for dataset with n= ' + str(n)) 
	# showing legend 
	plt.legend() 
	  
	# function to show the plot 
	plt.show()
	


def Graph_error(train_error,test_error,n):
	x=range(1,10)
	# plotting the line 1 points  
	plt.plot(x, train_error, label = "train_error",color='green') 
	  
	# plotting the line 2 points  
	plt.plot(x, test_error, label = "test_error",color='red') 
	  
	# naming the x axis 
	plt.xlabel('n') 
	# naming the y axis 
	plt.ylabel('Error') 
	# giving a title to my graph 
	plt.title('Train_error vs Test_error '+ ' for dataset of size ' + str(n)) 
	  
	# show a legend on the plot 
	plt.legend() 
	  
	# function to show the plot 
	plt.show() 


def Graph_RMSE(x,y,n,q):
	# plotting the line 1 points  
	plt.plot(x, y) 
	  
	# naming the x axis 
	plt.xlabel('Learning Rate') 
	# naming the y axis 
	plt.ylabel('RMSE') 
	# giving a title to my graph 
	plt.title('RMSE vs Learning Rate for Gradient descent method ' + str(n) + ' for n = ' + str(q)) 
	  
	# show a legend on the plot 
	plt.legend() 
	  
	# function to show the plot 
	plt.show() 






def gradientDescent(X, y, theta, learning_rate, iteration):
	sizeX=y.shape[0]

	for i in range(iteration):
		k=np.dot(X,theta)-y
		theta=theta-(learning_rate*1.0/sizeX)*np.dot(X.T,k)

	return theta	



def gradientDescent_2(X, y, theta, learning_rate, iteration):
	sizeX=y.shape[0]

	for d in range(iteration):
		k=np.dot(X,theta)-y
		for i in range (0,X.shape[1]) :
			temp=0
			for p in range(0,sizeX):
				if k[p,0]<0 :
					temp=temp-learning_rate*X[p,i]
				else:
					temp=temp+learning_rate*X[p,i]

			theta[i,0]=theta[i,0]-temp/(2*X.shape[0])

	return theta



def gradientDescent_3(X, y, theta, learning_rate, iteration):
	sizeX=y.shape[0]

	for i in range(iteration):
		k=np.dot(X,theta)-y
		k=k**3
		theta=theta-(learning_rate*2.0/sizeX)*np.dot(X.T,k)

	return theta


def cost_compute(X, y, theta):
	sizeX,sizeY=y.shape

	t=np.dot(X,theta)
	t=t-y
	t=t**2

	J=(1/(2*sizeX))*sum(t);

	return J




def generate_data(n,m,x_train,y_train,x_test,y_test):

	theta=np.zeros((m,1))

	temp=[[i] for i in y_train]
	Y=np.array(temp)
	#print(Y)
	
	# print(x_test)
	# print("geeg")


	temp=ones((len(x_train),1))
	X=np.array(temp)


	for i in range(1,m):
		temp=[[x**i] for x in x_train]
		X=np.hstack((X,temp))
	
	X_test=np.array(ones((len(x_test),1)))
	
	for i in range(1,m):
		temp=[[x**i] for x in x_test]
		X_test=np.hstack((X_test,temp))
	
	temp=[[i] for i in y_test]
	Y_test=np.array(temp)	

	# print(X_test)

	
	return [X,Y,theta,X_test,Y_test]


def Test_it(n,o,alpha):
	x=np.random.uniform(0,1,n)
	y=[]

	for i in range(0,n):
		y.append(math.sin(2*math.pi*x[i])+np.random.normal(0, 0.3))



	rand=rd.sample(range(n), 2*n//10)

	x_test=[x[i] for i in rand]
	y_test=[y[i] for i in rand]

	x_train=[x[i] for i in range(0,n) if i not in rand]
	y_train=[y[i] for i in range(0,n) if i not in rand]

	

	y_predicted_test=[]
	y_predicted_train=[]
	square_error_test=[]
	square_error_train=[]
	for m in range(2,11):
		[X,Y,theta,X_test,Y_test]=generate_data(n,m,x_train,y_train,x_test,y_test)

		


		if o==1:
			theta=gradientDescent(X,Y,theta,alpha,10000)
		if o==2:
			theta=gradientDescent_2(X,Y,theta,alpha,10000)
		if o==3:
			theta=gradientDescent_3(X,Y,theta,alpha,10000)
		#print(X)
		#print(Y)
		

		#print(cost_compute(X_test,Y_test,theta))
		t=np.dot(X_test,theta)
		y_predicted_test.append(t) # Saving the values of y_predicted on test data to plot the graph later
		y_predicted_train.append(np.dot(X,theta))
		


		square_error_test.append(2*cost_compute(X_test,Y_test,theta))
		square_error_train.append(2*cost_compute(X,Y,theta))
		
	




	return [x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]





Train_error=[]
Test_error=[]


#For n=10



[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(10,1,0.05)


X_original_train=x_train
Y_original_train=y_train
X_original_test=x_test
Y_original_test=y_test


#plotting graph now



Graph(x,y,10)

for i in range(0,9):
	Graph_2(x_train,y_train,y_predicted_train[i],i+1,10)

#Part 2b
Graph_error(square_error_train,square_error_test,10)

Train_error.append(square_error_train)
Test_error.append(square_error_test)



##########################################################################
#For n=100

[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(100,1,0.05)

#plotting graph now


Graph(x,y,100)

for i in range(0,9):
	Graph_2(x_train,y_train,y_predicted_train[i],i+1,100)

#Part 2b
Graph_error(square_error_train,square_error_test,100)


Train_error.append(square_error_train)
Test_error.append(square_error_test)

##########################################################################
#For n=1000

[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(1000,1,0.05)

#plotting graph now


Graph(x,y,1000)

for i in range(0,9):
	Graph_2(x_train,y_train,y_predicted_train[i],i+1,1000)

#Part 2b
Graph_error(square_error_train,square_error_test,1000)


Train_error.append(square_error_train)
Test_error.append(square_error_test)



#########################################################################
#For n=10000

[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(10000,1,0.05)

#plotting graph now


Graph(x,y,10000)

for i in range(0,9):
	Graph_2(x_train,y_train,y_predicted_train[i],i+1,10000)

#Part 2b
Graph_error(square_error_train,square_error_test,10000)


Train_error.append(square_error_train)
Test_error.append(square_error_test)


#############################################################################

x=[10,100,1000,10000]

for i in range(0,9):
	y_train=[]
	y_test=[]
	for k in range(0,4):
		y_train.append(Train_error[k][i])
		y_test.append(Test_error[k][i])
	Graph_3(x,y_train,y_test,i+1)



##############################################################################
# 4 A part
# For Mean absolute error function


#For n=10

[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(10,2,0.05)


#plotting graph now


Graph(x,y,10)

for i in range(0,9):
	Graph_2(x_train,y_train,y_predicted_train[i],i+1,10)


#Part 2b
Graph_error(square_error_train,square_error_test,10) 




##############################################################################
# 4 B part
# For Fourth power error function


#For n=10

[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(10,3,0.05)


#plotting graph now


Graph(x,y,10)

for i in range(0,9):
	Graph_2(x_train,y_train,y_predicted_train[i],i+1,10)


#Part 2b
Graph_error(square_error_train,square_error_test,10) 


##############################################################################




for q in range(0,9):
	alpha=[0.025,0.05,0.1,0.2,0.5]

	RMSE_1=[]
	RMSE_2=[]
	RMSE_3=[]

	for x in range(0,5):
		[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(10,1,x)
		RMSE_1.append(math.sqrt(square_error_test[q]))


	for x in range(0,5):
		[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(10,2,x)
		RMSE_2.append(math.sqrt(square_error_test[q]))

	for x in range(0,5):
		[x,y,x_train,y_train,x_test,y_test,y_predicted_test,y_predicted_train,square_error_test,square_error_train]=Test_it(10,3,x)
		RMSE_3.append(math.sqrt(square_error_test[q]))




	Graph_RMSE(alpha,RMSE_1,1,q)
	Graph_RMSE(alpha,RMSE_2,2,q)
	Graph_RMSE(alpha,RMSE_3,3,q)
	


























