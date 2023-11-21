import numpy as np
from sklearn.neural_network import MLPClassifier

#########################################################################
# NN
#

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.
    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)
    theta2: array_like
        Weights for the second layer in the neural network.
        It has shape (output layer size x 2nd hidden layer size)
    X : array_like
        The image inputs having shape (number of examples x image dimensions).
    Return
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    # Add a column of ones to X (bias term)
    m = X.shape[0]
    X1s = np.hstack([np.ones((m, 1)), X])
    # Calculate activations in the second layer
    # Add a column of ones to a2 (bias term)
    a2 = np.hstack([np.ones((m, 1)),  sigmoid(np.dot(X1s, theta1.T))])

    # Calculate activations in the output layer
    h = sigmoid(np.dot(a2, theta2.T))
    
    # Find the index of the maximum value in each row (predicted label)
    p = np.argmax(h, axis=1)
    p = p + 1  # Adjust for 1-based indexing (labels are 1 to 10)
    return p
