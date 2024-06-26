import numpy as np

#Step Funtion
def step_funtion(x):
    y = x > 0 
    return y.astype(np.int) #return 0 if the value is negative and 1 if the value is positive

#Sigmoid Function
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#softmax Function
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

#Relu Function
def relu(x):
    return np.maximum(0,x) #return 0 if the value is negative and return the value itself if it's greater than 0

#Sum Squared Error Function
def sum_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

#Cross-Entropy Error Function (output should be one-hot expression)
def cross_entropy_error(y,t):
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

#Numerical Gradient Function
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad
        
#im2col関数
def im2col(input_data, filter_h, filter_w, stride = 1, pad = 0):
    N,C,H,W = input_data.shape
    out_h = (H* 2*pad -filter_h)//stride + 1
    out_w = (W* 2*pad -filter_w)//stride + 1
    
    img = np.pad(input_data,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    col = np.zeros_like(N,C,filter_h,filter_w,out_h,out_w)
    
    for y in range(filter_h):
        y_max = y+stride*out_h
        for x in range(filter_h):
            x_max = x+stride*out_h
            col[:,:,y,x,:,:] = img[:,:,y:y_max:stride,x:x_max:stride]
    
    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col

#col2im関数
def col2im(col,input_shape,filter_h,filter_w,stride = 1, pad = 0):
    N,C,H,W = input_shape
    out_h = (H* 2*pad -filter_h)//stride + 1
    out_w = (W* 2*pad -filter_w)//stride + 1
    col = col.reshape(N,out_h,out_w,C,filter_h,filter_w).transpose(0,3,4,5,1,2)
    
    img = np.zeros(N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1)
    for y in range(filter_h):
        y_max = y+ stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:,:,y:y_max:stride,x:x_max:stride] += col[:,:,y,x,:,:]
    
    return img[:,:,pad:H+pad,pad:W+pad]




