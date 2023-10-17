import numpy as np
import copy
import math




#########################################################################
# Cost function
 """Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
"""
def compute_cost(x, y, w, b):
    total_cost=0
    for i in range(len(x)):
        J=((w + b*x[i]) - y[i]**i)**2
        total_cost= total_cost + J
    
    
    total_cost= (1/2*len(x))*total_cost
    return total_cost


#########################################################################
# Gradient function
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
#
def compute_gradient(x, y, w, b):
    
    return dj_dw, dj_db


#########################################################################
# gradient descent
"""
    Performs batch gradient descent to learn theta. Updates theta by taking num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    theta=0
    w=0
    b=0
    for i in range(num_iters):
        w=cost_function(x,y,theta)
        aux=gradient_function(x,y,theta,alpha)
        theta=aux
        
    
    return w, b, J_history
