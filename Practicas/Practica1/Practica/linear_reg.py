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
     m = len(y)  # Nº examples

    # Calcular la hipótesis
    h = w * x + b # linear regresion so f(x)= w*x + b

    #  (MSE - Mean Squared Error)
    error = h-y
    squared_error = error**2
    total_cost = (1 / (2*m)) * np.sum(squared_error)

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
    J_history=np.zeros(num_iters)
    
    w=w_in
    b=_b_in
    
    m=len(y) # Nº examples
    
    for i in range(num_iters):
        dj_dw, dj_db=  compute_gradient(x,y,w,b)
        w-=alpha * dj_dw
        b-=alpha * dj_db
        J_history[i]= compute_cost(x,y,w,b)
        
    
    return w, b, J_history
