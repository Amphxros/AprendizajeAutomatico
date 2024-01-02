import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

num_states = 2
num_actions =  7

# Hiperparámetros
learning_rate = 0.001
discount_factor = 0.99
exploration_prob = 0.2
memory_size = 10000
batch_size = 32

model = Sequential()
model.add(Dense(64, input_dim=num_states, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_actions, activation='linear'))
model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

# Inicializa la memoria de reproducción
memory = []