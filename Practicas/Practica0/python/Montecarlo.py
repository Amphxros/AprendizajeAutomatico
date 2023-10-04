# Practica 0 Python
import numpy as np
import matplotlib.pyplot as plt # drawing library (same one i used in PEV)
import time as t 

# Funcion
def cuadrado(x):
    return -(x * x)

#Returns whether is f(x) is lower than the point
def is_behind(fun, x, y):
    Y = fun(x)
    return y < Y 

# Generates random points
def gen_points(min_, max_, num_puntos):
    a = np.zeros(num_puntos)
    for i in range(num_puntos):
        a[i] = np.random.uniform(min_, max_, 1)

    return a

# Draws the graphic
def draw_graphic(x, y, ry):
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(x, y, color='black') # draws line
    plt.scatter(x[np.where(ry > y)], ry[np.where(ry > y)], marker='x', color='red') # draws a x when is out
    plt.scatter(x[(ry <= y)], ry[(ry <= y)], marker='o', color='blue') # draws an o when it's in
    plt.show()
