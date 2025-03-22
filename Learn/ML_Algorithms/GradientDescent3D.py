import numpy as np
import matplotlib.pyplot as plt

def z_function_1(x, y):
    return np.sin(5*x) * np.cos(5*y) / 5

def z_gradient_1(x, y):
    return np.cos(5*x) * np.cos(5*y), -np.sin(5*x) * np.sin(5*y)

def z_function_2(x, y):
    return x**2 + y ** 2

def z_gradient_2(x, y):
    return 2*x, 2*y

x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)

X, Y = np.meshgrid(x, y)

Z = z_function_1(X, Y)

current_pos = (0.7, 0.4, z_function_1(0.7, 0.4))

learning_rate = 0.01
max_steps = 1000
current_steps = 0

gd_model = plt.subplot(projection="3d", computed_zorder=False)

while(current_steps < max_steps):

    X_derivative, Y_derivative = z_gradient_1(current_pos[0], current_pos[1])
    X_step_size, Y_step_size = X_derivative * learning_rate, Y_derivative * learning_rate

    X_new, Y_new = current_pos[0] - X_step_size, current_pos[1] - Y_step_size
    current_pos = (X_new, Y_new, z_function_1(X_new, Y_new))
    current_steps+=1

    gd_model.plot_surface(X, Y, Z, cmap="viridis", zorder=0)
    gd_model.scatter(current_pos[0], current_pos[1], current_pos[2], c="red", zorder=1)

    x = np.arange(0, 10, 0.1)
    y = current_pos[0] * x + current_pos[1]
    plt.pause(0.01)
    gd_model.clear()