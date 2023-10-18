import numpy as np
import copy
import math

#########################################################################
# Cost function
def compute_cost(x, y, w, b):
    m=len(y)
    error = (w*x+b)-y
    squared_error=error**2
    total_cost = (1 / (2*m)) * np.sum(squared_error)
    return total_cost


#########################################################################
# Gradient function
def compute_gradient(x, y, w, b):
    
    m = len(y) # Nº examples
    
    # hipotesis
    h = w*x + b # linear regresion so f(x)= w*x + b

    # gradient w (dj_dw)
    dj_dw = (1/m) * np.sum((h-y) * x)

    # gradient b (dj_db)
    dj_db = (1/m) * np.sum(h-y)

    
    return dj_dw, dj_db


#########################################################################
# gradient descent
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history=np.zeros(num_iters)
    
    w=w_in
    b=b_in
    
    m=len(y) # Nº examples
    
    for i in range(num_iters):
        dj_dw, dj_db=  compute_gradient(x,y,w,b)
        w-=alpha * dj_dw
        b-=alpha * dj_db
        J_history[i]= compute_cost(x,y,w,b)
        
    
    return w, b, J_history
