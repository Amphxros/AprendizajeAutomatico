import numpy as np

#########################################################################
# Forward Propagation

def feedForward(theta1, theta2, X):
	"""
	Funcion de propagación hacia delante para una red de 3 capas 

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
	a3 : array_like
		Salida de la red 			[numero de ejemplos x numero de neuronas de salida]

	"""
	m = len(X) # Numero de ejemplos

	# La capa de entrada será de 400 neuronas (20x20 pixeles)
	# La capa oculta de 25 neuronas
	a1 = np.hstack([np.ones((m,1)), X])	# Añade columna de 1 (bias)	[5000, 401]
	z2 = np.dot(a1, theta1.T)			# Aplica pesos a la capa 	[5000, 25]
	a2 = sigmoid(z2)					# Aplica funcion de activacion

	# Repite para la capa de salida de 10 neuronas
	a2 = np.hstack([np.ones((m,1)), a2])# Añade columna de 1 (bias)	[5000, 26]
	z3 = np.dot(a2, theta2.T)			# Aplica pesos a la capa 	[5000, 10]
	a3 = sigmoid(z3)
	return a3

#########################################################################
# Cost function

def cost(theta1, theta2, X, y, lambda_):
	"""
	Funcion de coste para una red de 3 capas 

	Parametros
	----------
	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa oculta x capa de entrada]

	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa de salida x capa oculta]

	X : array_like
		Entradas de la red 			[numero de ejemplos x numero de dimensiones]

	y : array_like
		Etiquetas ya classificadas en One-hot encoding [numero de ejemplos x Numero de etiquetas]

	lambda : float
		Parametro de regularización. 

	Returns
	-------
	J : float
		Valor total de la función de coste 

	"""

	# Para calcular el coste se hace un feedforward inicial

	a3 = feedForward(theta1, theta2, X) # Aplica la red neuronal a los datos, devuelve las 10 neuronas de salida

	m = len(X) # Numero de ejemplos

	# Funcion de coste aplicada a las neuronas (batiburrillo de formulas de los apuntes)
	J = (-1 / m) * np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))

	# Aplicamos L2 (Mean Square Error) [Tochaco de los apuntes]
	J += (lambda_ / (2 * m)) * (np.sum(np.square(theta1)) + np.sum(np.square(theta2)))

	return J

#########################################################################
# Backwards Propagation

def backProp(theta1, theta2, X, y, lambda_):
	"""
	Funcion que calcula la gradiente mediante retropropagacion 

	Parametros
	----------
	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa oculta x capa de entrada]

	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa de salida x capa oculta]

	X : array_like
		Entradas de la red 			[numero de ejemplos x numero de dimensiones]

	y : array_like
		Etiquetas ya classificadas en One-hot encoding [numero de ejemplos x Numero de etiquetas]

	lambda : float
		Parametro de regularización. 

	Returns
	-------
	J : float
		Valor total de la función de coste 

	grad1 : array_like
		Gradient of the cost function with respect to weights
		for the first layer in the neural network, theta1.
		It has shape (2nd hidden layer size x input size + 1)

	grad2 : array_like
		Gradient of the cost function with respect to weights
		for the second layer in the neural network, theta2.
		It has shape (output layer size x 2nd hidden layer size + 1)
	"""

	# No podemos usar el método feedForward porque necesitamos los valores intermedios
	m = len(X) # Numero de ejemplos

	a1 = np.hstack([np.ones((m,1)), X])	# Añade columna de 1 (bias)	[5000, 401]
	z2 = np.dot(a1, theta1.T)			# Aplica pesos a la capa 	[5000, 25]
	a2 = sigmoid(z2)					# Aplica funcion de activacion

	# Repite para la capa de salida de 10 neuronas
	a2 = np.hstack([np.ones((m,1)), a2])# Añade columna de 1 (bias)	[5000, 26]
	z3 = np.dot(a2, theta2.T)			# Aplica pesos a la capa 	[5000, 10]
	a3 = sigmoid(z3)


	# Derivada parcial del error de la ultima capa con las etiquetas
	#delta3 = (a3 - y) * (a3 * (1 - a3))	
	delta3 = a3 - y

	# Derivada parcial de la segunda capa respecto al errpr de la tercera
	#delta2 = np.dot(delta3, theta2[:,1:]) * (a2[:,1:] * (1 - a2[:,1:]))
	delta2 = np.dot(delta3, theta2[:,1:]) * (a2[:,1:] * (1 - a2[:,1:]))

	# Gradientes (Recalculado de los pesos)
	# Esto es la retropropagacion de los errores
	grad1 = (1/m)*np.dot(delta2.T, a1)
	grad2 = (1/m)*np.dot(delta3.T, a2)

	J = cost(theta1, theta2, X, y, lambda_)
	
	return (J, grad1, grad2)

#########################################################################
# entrenamiento

def training (X, Y, hidden_size, output_size, epsilon, alpha, lambda_, epochs):
	"""
	Funcion que entrena una red neuronal de 3 capas

	Parametros
	----------
	X : array_like
		Entradas de la red 			[numero de ejemplos x numero de dimensiones]

	Y : array_like
		Etiquetas ya classificadas en One-hot encoding [numero de ejemplos x Numero de etiquetas]

	hidden_size : int
		Numero de neuronas en la capa oculta

	output_size : int
		Numero de neuronas en la capa de salida

	epsilon : float
		Valor de la inicialización aleatoria de los pesos

	alpha : float
		Valor de la tasa de aprendizaje

	lambda : float
		Parametro de regularización. 

	epochs : int
		Numero de iteraciones para el entrenamiento

	Returns
	-------
	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa oculta x capa de entrada]

	theta1 : array_like
		Pesos de la promera capa	[tamaño de capa de salida x capa oculta]

	"""

	# Inicialización de los pesos semi-aleatoria
	theta1 = np.random.rand(hidden_size, X.shape[1] + 1) * 2 * epsilon - epsilon
	theta2 = np.random.rand(output_size, hidden_size + 1) * 2 * epsilon - epsilon

	# Entrenamiento
	for epoch in range(epochs):
		J, grad1, grad2 = backProp(theta1, theta2, X, Y, lambda_)

		m = len(X)
		# Actualización de los pesos
		theta1 -= alpha * grad1
		theta2 -= alpha * grad2

		if epoch % 100 == 0:
			print(f'Epoch {epoch}, Cost: {J}')

	# Esto seria la red entreada ya
	return theta1, theta2

#########################################################################
# sigmoid function

def sigmoid(z):
	clipped_z = np.clip(z, -1000, 1000)
	return (1 / (1 + np.exp(-clipped_z)))

#########################################################################
# predict function

def predict(theta1, theta2, X):

	print(X.shape)
	output = feedForward(theta1,theta2,X) # Aplica la red neuronal a los datos, devuelve las 10 neuronas de salida
	print(output.shape)
	
	# Obtén el índice del valor máximo en cada fila (one-hot decoding)
	decoded_output = np.argmax(output, axis=1)
    
    # Reshape el array para que tenga forma [5000, 1]
	reshaped_output = decoded_output.reshape(-1, 1)
    
	return reshaped_output