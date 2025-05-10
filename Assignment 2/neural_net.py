import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Dataset preparation
data = pd.read_csv("AI1104-final-project-Q1-dataset.csv")
training_x1 = data["x1"].to_numpy()
training_x2 = data["x2"].to_numpy()
training_y = data["y"].to_numpy()
n = len(training_x1)
training_x = np.array([[x1,x2,1] for x1,x2 in zip(training_x1,training_x2)])

# Neural network preliminaries
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 2 input nodes , 3 hidden nodes and 1 output node

# Definitions and initialisations

np.random.seed(123)
def grad_sigmoid(x):
    ex = np.exp(x)
    return ex/((1+ex)**2)

weights1 = []
for _ in range(9):
    num = np.random.rand()
    weights1.append(2*num - 1)
weights1 = np.array(weights1).reshape(3,3)

weights2 = []
for _ in range(4):
    num = np.random.rand()
    weights2.append(2*num-1)
weights2 = np.array(weights2).reshape(4,1)

learning_rate = 0.05
max_epochs = 100
training_error = []
epochs = []
# Training the neural network
def train_neural_network():
    global weights1,weights2,training_x
    for epoch in range(max_epochs):
        # Forward pass
        X = training_x
        H = np.matmul(X,weights1)
        Z = sigmoid(H)
        Z = np.hstack((Z, np.ones((n, 1))))
        O = np.matmul(Z,weights2)
        y_hat = sigmoid(O)
        loss = mean_squared_error(training_y,y_hat)
        training_error.append(loss)
        epochs.append(epoch)


        # Backpropagation
        # Delta for output layer
        delta_output = 2 * (y_hat - training_y.reshape(-1, 1)) * y_hat * (1 - y_hat)  # shape: (n, 1)

        # Gradient of loss w.r.t weights2
        gradient2 = np.matmul(Z.T, delta_output)  # shape: (4, 1)
        gradient2 /= len(training_y)

        delta_hidden = (delta_output @ weights2[:-1].T) * grad_sigmoid(H)
        gradient1 = X.T @ delta_hidden
        gradient1 /= len(training_y)


        weights2 = weights2 - learning_rate*gradient2
        weights1 = weights1 - learning_rate*gradient1

train_neural_network()
plt.plot(epochs,training_error,color='blue',label='Loss vs Epoch')
plt.xlabel('Epochs completed')
plt.ylabel('Training error')
plt.title('Training error vs Epoch')
plt.show()

