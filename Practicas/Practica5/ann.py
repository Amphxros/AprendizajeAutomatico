import numpy as np
from sklearn.neural_network import MLPClassifier

def initialize_params(input_size, hidden_size, output_size):
    epsilon_init = 0.12
    theta1 = np.random.rand(hidden_size, input_size + 1) * 2 * epsilon_init - epsilon_init
    theta2 = np.random.rand(output_size, hidden_size + 1) * 2 * epsilon_init - epsilon_init
    return theta1, theta2
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost(theta1, theta2, X, y, lambda_):
   """Compute cost for 2-layer neural network. Parameters
   ----------
    theta1 : array_like
    Weights for the first layer in the neural network.
    It has shape (2nd hidden layer size x input size + 1)
    theta2: array_like
    Weights for the second layer in the neural network.
    It has shape (output layer size x 2nd hidden layer size + 1)
    X : array_like
    The inputs having shape (number of examples x number of dimensions).
    y : array_like
    2
    1-hot encoding of labels for the input, having shape
    (number of examples x number of labels).
    lambda_ : float
    The regularization parameter.
    Returns
    -------
    J : float
    The computed value for the cost function.
    """
    m = len(y)
    ones = np.ones((m, 1))
    a1 = np.hstack((ones, X))
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
              
    ## coste
    J=(-1/m)* np.sum(y*np.log(h) + (1 - y) * np.log(1-h))
    ## regularizacion
    reg= (lambda_/(2*m))* np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]) 
    J=J+reg
    return J


def backdrop(theta1. theta2, X,y,lambda_):
    m=len(y)
    one=np.ones(m,1)
    a1 = np.hstack((ones, X))
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    #cost
    J=cost(theta1, theta2, X, y, lambda_)
    # Compute gradients
    grad1 = np.dot(delta2.T, a1) / m
    grad2 = np.dot(delta3.T, a2) / m

    # Regularization
    grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
    grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]
    # back
    delta3=h-y
    delta2=np.dot(delta3, theta2[:, 1:]) * sigmoid(z2) * (1 - sigmoid(z2))
    
    
    return J, grad1, grad2
def train_neural_network(X,y,input_size,hidden,output_size,lambda_,alpha,num_iterations):
    m=X.shape()
    one=np.ones(m,1)
    # One-Hot Encoding
    y_onehot = np.eye(output_size)[y.flatten() - 1]
    # Initialize parameters
    theta1, theta2 = initialize_params(input_size, hidden_size, output_size)
    for i in range(num_iterations):
         # Flatten parameters for optimization
        params = np.concatenate((theta1.flatten(), theta2.flatten()))
         # Define cost and gradient function for optimization
        cost_func = lambda p: backprop(np.reshape(p[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)),
                                       np.reshape(p[hidden_size * (input_size + 1):], (output_size, hidden_size + 1)),
                                       X, y_onehot, lambda_)[0]

        grad_func = lambda p: np.concatenate((backprop(np.reshape(p[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)),
                                                       np.reshape(p[hidden_size * (input_size + 1):], (output_size, hidden_size + 1)),
                                                       X, y_onehot, lambda_)[1].flatten(),
                                              backprop(np.reshape(p[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)),
                                                       np.reshape(p[hidden_size * (input_size + 1):], (output_size, hidden_size + 1)),
                                                       X, y_onehot, lambda_)[2].flatten()))
    
     # Use scipy's minimize function for optimization
    result = minimize(fun=cost_func, x0=params, jac=grad_func, method='TNC', options={'maxiter': 100, 'disp': True})

    # Reshape optimized parameters back to matrices
    theta1 = np.reshape(result.x[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1))
    theta2 = np.reshape(result.x[hidden_size * (input_size + 1):], (output_size, hidden_size + 1))

    return theta1, theta2
    
def predict(theta1, theta2, X):
    m = len(X)
    ones = np.ones((m, 1))
    a1 = np.hstack((ones, X))
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    # Return the index (starting from 1) of the maximum value in each row
    return np.argmax(h, axis=1) + 1