import numpy as np

#Step Funtion
def step_funtion(x):
    y = x > 0 
    return y.astype(np.int) #return 0 if the value is negative and 1 if the value is positive

#Sigmoid Function
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#Relu Function
def relu(x):
    return np.maximum(0,x) #return 0 if the value is negative and return the value itself if it's greater than 0


