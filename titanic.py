# %% [code] {"execution":{"iopub.status.busy":"2023-09-25T06:47:57.000187Z","iopub.execute_input":"2023-09-25T06:47:57.000770Z","iopub.status.idle":"2023-09-25T06:47:57.467320Z","shell.execute_reply.started":"2023-09-25T06:47:57.000735Z","shell.execute_reply":"2023-09-25T06:47:57.466184Z"}}
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/titanic/train.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:24:45.219504Z","iopub.execute_input":"2023-09-24T10:24:45.219857Z","iopub.status.idle":"2023-09-24T10:24:45.229941Z","shell.execute_reply.started":"2023-09-24T10:24:45.219826Z","shell.execute_reply":"2023-09-24T10:24:45.228715Z"}}
%config Completer.use_jedi = False

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T11:16:38.095160Z","iopub.execute_input":"2023-09-24T11:16:38.095621Z","iopub.status.idle":"2023-09-24T11:16:38.108386Z","shell.execute_reply.started":"2023-09-24T11:16:38.095585Z","shell.execute_reply":"2023-09-24T11:16:38.106648Z"}}
data = np.array(data)
m, n = data.shape
#np.random.shuffle(data) # shuffle before splitting into dev and training sets

def fixData(Z):
    for arr in Z:
        if arr[1] == "male":
            arr[1] = 0
        else:
            arr[1] = 1
    for el in X_train:
        if np.isnan(el[2]):
            el[2] = 30 #average age
    return Z

data_dev = data[0:5]
Y_dev = data_dev[:, 1]
X_dev = np.delete(data_dev, [0,1,3,6,7,8,10,11], axis=1)
X_dev = fixData(X_dev).T

data_train = data[0:5]
names_train = data_train[:,3]
Y_train = data_train[:, 1]
X_train = np.delete(data_train, [0,1,3,6,7,8,10,11], axis=1)
X_train = fixData(X_train).T

print(Y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:48:57.161022Z","iopub.execute_input":"2023-09-24T10:48:57.161444Z","iopub.status.idle":"2023-09-24T10:48:57.176744Z","shell.execute_reply.started":"2023-09-24T10:48:57.161412Z","shell.execute_reply":"2023-09-24T10:48:57.175822Z"}}
def init_params():
    W1 = np.random.rand(2, 4) - 0.5
    b1 = np.random.rand(2, 1) - 0.5
    W2 = np.random.rand(4, 2) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_deriv(Z):
    return Z * (1 - Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    #dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dZ1 = W2.dot(dZ2)*sigmoid_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:46:58.834443Z","iopub.execute_input":"2023-09-24T10:46:58.834890Z","iopub.status.idle":"2023-09-24T10:46:58.844758Z","shell.execute_reply.started":"2023-09-24T10:46:58.834856Z","shell.execute_reply":"2023-09-24T10:46:58.843775Z"}}
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T11:20:42.961113Z","iopub.execute_input":"2023-09-24T11:20:42.961532Z","iopub.status.idle":"2023-09-24T11:20:42.975657Z","shell.execute_reply.started":"2023-09-24T11:20:42.961500Z","shell.execute_reply":"2023-09-24T11:20:42.974772Z"}}
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def init_params():
    W1 = np.random.rand(2, 4) - 0.5
    b1 = np.random.rand(2, 1) - 0.5
    W2 = np.random.rand(2, 2) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

X_train = np.array([[3, 1, 3, 1, 3],
                    [0, 1, 1, 1, 0],
                    [22.0, 38.0, 26.0, 35.0, 35.0],
                    [7.25, 71.2833, 7.925, 53.1, 8.05]])

W1, b1, W2, b2 = init_params()
Z1 = W1.dot(X_train)+b1
A1 = sigmoid(Z1)
Z2 = W2.dot(A1)+b2
A2 = softmax(Z2)

one_hot_Y = np.zeros((Y_train.size, Y_train.max() + 1))
print(one_hot_Y)
print(Y_train.size)




# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T11:14:58.079419Z","iopub.execute_input":"2023-09-24T11:14:58.079888Z","iopub.status.idle":"2023-09-24T11:14:58.165614Z","shell.execute_reply.started":"2023-09-24T11:14:58.079854Z","shell.execute_reply":"2023-09-24T11:14:58.164362Z"}}
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 100)
