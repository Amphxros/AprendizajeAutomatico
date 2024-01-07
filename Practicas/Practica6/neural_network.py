import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar el dataset
archivos= os.listdir('data') # movem
num_archivos = len(archivos)
numero_de_archivos = len(archivos)
print(f'Número de archivos en el directorio: {numero_de_archivos}')
X=[]
Y=[]
# parsear datos ... 
for archivo in archivos:
        print(archivo) # nombre para refernciarnos
        df= pd.read_csv('data/' + archivo)
        
        x = df[['ray1', 'ray2', 'kartx', 'karty', 'kartz']]  # ajusta las características según tu dataset
        y = df['action']
        
        X.append(x)
        Y.append(y)       


# Seleccionar características y objetivo
# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)  # ajusta el número de vecinos según sea necesario
knn.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = knn.predict(X_test)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()