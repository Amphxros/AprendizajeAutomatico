import numpy as np

#########################################################################
# NN
#
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

	return p
