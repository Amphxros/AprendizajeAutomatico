import numpy as np

#########################################################################
# NN
#

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(theta1, theta2, X):
    """
    Aplica una prediccion a X dada una red entrenada en formato de pesos theta1 y theta2.
    Parameters
    ----------
	Parametros
	----------
	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa oculta x capa de entrada]

	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa de salida x capa oculta]

	X : array_like
		Entradas de la red 			[numero de ejemplos x numero de dimensiones]

	Devuelve
	-------
	p : array_like
		Salida de la red 			[numero de ejemplos x numero de neuronas de salida]
    """
    m = X.shape[0]	# Numero de ejemplos
    
    X1s = np.hstack([np.ones((m, 1)), X])	# Añade columna de 1 (bias)

    a2 = sigmoid(np.dot(X1s, theta1.T))	    # Capa oculta, resultadod e aplicar pesos a la entrada
    a2 = np.hstack([np.ones((m, 1)), a2])	# Añade columna de 1 (bias)

    h = sigmoid(np.dot(a2, theta2.T))		# Capa de salida, resultado de aplicar pesos a la capa oculta

    p = np.argmax(h, axis=1) 				# "deshace" el one-hot encoding

    return p