import numpy as np
from sklearn.neural_network import MLPClassifier

def cost(theta1, theta2, X, y, lambda_):
	"""
	Compute cost for 2-layer neural network. 

	Parameters
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
		1-hot encoding of labels for the input, having shape 
		(number of examples x number of labels).

	lambda_ : float
		The regularization parameter. 

	Returns
	-------
	J : float
		The computed value for the cost function. 

	"""
       """
    Compute cost for 2-layer neural network.
    
    Parameters
    ----------
    theta1 : array_like
        Pesos para la primera capa en la red neuronal.
        Tiene forma (tamaño de la 2ª capa oculta x tamaño de la entrada + 1).
    theta2 : array_like
        Pesos para la segunda capa en la red neuronal.
        Tiene forma (tamaño de la capa de salida x tamaño de la 2ª capa oculta + 1).
    X : array_like
        Las entradas con forma (número de ejemplos x número de dimensiones).
    y : array_like
        Codificación one-hot de las etiquetas para la entrada, con forma
        (número de ejemplos x número de etiquetas).
    lambda_ : float
        El parámetro de regularización.
    
    Returns
    -------
    J : float
        El valor calculado para la función de costo.
    """
    m = len(X)  # Número de ejemplos
    
    # Propagación hacia adelante
    a1 = np.insert(X, 0, values=1, axis=1)  # Agregar sesgo a las entradas
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1)  # Agregar sesgo a la capa oculta
    z3 = a2.dot(theta2.T)
    h = sigmoid(z3)  # Salida de la red
    
    # Calcular el costo sin regularización
    cost_unreg = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    # Calcular la regularización (ignorando los sesgos)
    reg_term = (lambda_ / (2 * m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    
    # Costo total con regularización
    J = cost_unreg + reg_term
	return J



def backprop(theta1, theta2, X, y, lambda_):
   """
    Compute cost and gradient for 2-layer neural network.

    Parameters
    ----------
    theta1 : array_like
        Pesos para la primera capa en la red neuronal.
        Tiene forma (tamaño de la 2ª capa oculta x tamaño de la entrada + 1).

    theta2 : array_like
        Pesos para la segunda capa en la red neuronal.
        Tiene forma (tamaño de la capa de salida x tamaño de la 2ª capa oculta + 1).

    X : array_like
        Las entradas con forma (número de ejemplos x número de dimensiones).

    y : array_like
        Codificación one-hot de las etiquetas para la entrada, con forma
        (número de ejemplos x número de etiquetas).

    lambda_ : float
        El parámetro de regularización.

    Returns
    -------
    J : float
        El valor calculado para la función de costo.

    grad1 : array_like
        Gradiente de la función de costo con respecto a los pesos
        para la primera capa en la red neuronal, theta1.
        Tiene forma (tamaño de la 2ª capa oculta x tamaño de la entrada + 1).

    grad2 : array_like
        Gradiente de la función de costo con respecto a los pesos
        para la segunda capa en la red neuronal, theta2.
        Tiene forma (tamaño de la capa de salida x tamaño de la 2ª capa oculta + 1).
    """

    m = len(X)  # Número de ejemplos

    # Propagación hacia adelante
    a1 = np.insert(X, 0, values=1, axis=1)  # Agregar sesgo a las entradas
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1)  # Agregar sesgo a la capa oculta
    z3 = a2.dot(theta2.T)
    h = sigmoid(z3)  # Salida de la red

    # Calcular errores de la capa de salida
    delta3 = h - y

    # Calcular errores de la capa oculta
    delta2 = (delta3.dot(theta2[:, 1:])) * sigmoid_gradient(z2)

    # Acumular gradientes
    Delta2 = delta3.T.dot(a2)
    Delta1 = delta2.T.dot(a1)

    # Calcular el costo sin regularización
    cost_unreg = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    # Calcular la regularización (ignorando los sesgos)
    reg_term = (lambda_ / (2 * m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))

    # Costo total con regularización
    J = cost_unreg + reg_term

    # Calcular gradientes con regularización
    grad1 = (1 / m) * Delta1 + (lambda_ / m) * np.hstack([np.zeros((theta1.shape[0], 1)), theta1[:, 1:]])
    grad2 = (1 / m) * Delta2 + (lambda_ / m) * np.hstack([np.zeros((theta2.shape[0], 1)), theta2[:, 1:]])

    return J, grad1, grad2

## forward from the last exercise 

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
    m = X.shape[0]
    X1s = np.hstack([np.ones((m, 1)), X])

    # Calculate activations in the second layer
    a2 = sigmoid(np.dot(X1s, theta1.T))

    # Add a column of ones to a2 (bias term)
    a2 = np.hstack([np.ones((m, 1)), a2])

    # Calculate activations in the output layer
    h = sigmoid(np.dot(a2, theta2.T))

    # Find the index of the maximum value in each row (predicted label)
    p = np.argmax(h, axis=1)

    return p  # Adjust for 1-based indexing (labels are 1 to 10)
