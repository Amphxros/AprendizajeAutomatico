# Practica 0 ejercicio de Montecarlo

# Montecarlo: calculo de pi con puntos aleatorios

import pandas as pd
import numpy as np

#Point: definicion de puntos en el espacio bidimensional
class Point:
    def __init__(self,x,y): 
        self.x=x
        self.y=y

class Montecarlo:
    def __init__(self,numX,numY):
        X=np.random.random((numX,numY))
        print(X) 
        
        
