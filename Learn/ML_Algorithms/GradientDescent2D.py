import numpy as np
import matplotlib.pyplot as plt

def y_function(x):
    return x**2

def y_derivative(x): # Ableitung
    return 2*x

x = np.arange(-100, 100, 0.1)
y = y_function(x) 

current_pos = (80, y_function(80))
max_steps = 1000
current_steps = 0
learning_rate = 0.01

while(current_steps < max_steps):
    step_size = y_derivative(current_pos[0]) * learning_rate
    current_pos = (current_pos[0] - step_size, y_function(current_pos[0] - step_size))
    current_steps+=1
    plt.plot(x, y)
    plt.scatter(current_pos[0], current_pos[1], c="red")
    plt.pause(0.01)
    plt.clf()
