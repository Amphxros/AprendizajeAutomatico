import numpy as np

def load_data():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    X = data[:,[-3,-2]]   # Tamaño (Columna 1)
    y = data[:,-1]   # Numero de habitaciones (Columna 2)
    return X, y

def load_data_multi():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    # [:,:] la coma separa filas de columnas, los doble puntos indican que seleccionan todo
    # thing = data[:,:-1] coge TODAS salvo la ultima columna
    # thing2 = data[:,-1:] coge TODAS salvo la primera columna
    # thing 3 = data[:,[3,4]] coge columnas específicas
    x = data[:,:-1] # Tamaño, Numero de habitaciones
    y = data[:,-1] # Precio
    return x, y
