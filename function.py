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

#Sum Squared Error Function
def sum_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

#Cross-Entropy Error Function (output should be one-hot expression)
def closs_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y+delta)) / batch_size

#Numerical Differentiation Funtion
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)







