import numpy as np
import copy
import math


#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """

    m = len(y) #N elems
    
    error = w * x + b #expected value to calculate
    
    # Mean Squared Error formula
    squared_error = (error-y) **2 
    
    # MSE = np.sum(squared_error)/(2*m)
    total_cost = (1/(2*m)) * np.sum(squared_error)
    
    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
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
    
    m = len(y)
    hypothesis = w * x + b

    dj_dw = np.sum((hypothesis-y)*x)/m
    
    dj_db = np.sum((hypothesis-y))/m
    
    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

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
    J_history = np.zeros(num_iters)
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db= compute_gradient(x,y,w,b)
        w-= alpha*dj_dw
        b-= alpha *dj_db
        J_history[i]= cost_function(x,y,w,b)
        

    return w, b, J_history
