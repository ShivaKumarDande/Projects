import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

No_features = 2

def dataset_creation():
    Ntrain = [1000]
    for Nsamples in Ntrain:
        data, labels = generateData(Nsamples)
#         stacking the data (3 X 100) and labels (1 X 100) to create a data set of size 100 X 4
        data_set = pd.DataFrame(np.hstack((data, labels)))
        plot3(data, labels, 'T')
#         stroing the egenrated data set in a .csv file
        filename = 'training_dataset' + str(Nsamples) + '.csv'
        data_set.to_csv(filename, index = False)

#     creating validation data sets
    NValidation = 10000
    data, labels = generateData(NValidation)
    plot3(data, labels, 'V')
    data_set = pd.DataFrame(np.hstack((data, labels)))
    filename = 'validation_dataset' + str(NValidation) +'.csv'
    data_set.to_csv(filename, index = False)

def generateData(N):
    priors = [0.5, 0.5] # priors should be a row vector
    labels = np.zeros((1, N))
    labels = (np.random.rand(N) >= priors[1]).astype(int)
    labels = np.array([int(-1) if (t == 0) else int(1) for t in labels])
    
    X = np.zeros(shape = [N, No_features])
    for i in range(N):
        if labels[i] == 1: 
            X[i, 0] = 4 * np.cos(np.random.uniform(-np.pi, np.pi))
            X[i, 1] = 4 * np.sin(np.random.uniform(-np.pi, np.pi)) 
        elif labels[i] == -1: 
            X[i, 0] = 2 * np.cos(np.random.uniform(-np.pi, np.pi))
            X[i, 1] = 2 * np.sin(np.random.uniform(-np.pi, np.pi)) 
        X[i, :] += np.random.multivariate_normal([0, 0], np.eye(2))
    labels = np.reshape(labels, (N,1))
    return X, labels

def plot3(data, labels, dtype):
#   from matplotlib import pyplot
#   import pylab
#   from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    plt.scatter(data[(labels.ravel() == -1), 0], data[(labels.ravel() == -1), 1], marker = 'o', color='b')
    plt.scatter(data[(labels.ravel() == 1),0], data[(labels.ravel() == 1), 1], marker = '*', color='r')
    plt.xlabel("x1")
    plt.ylabel("x2")
    if (dtype == 'T'):
        plt.title('Training Dataset')
    else:
        plt.title('Validation Dataset')
    plt.show()

def predict(data, e):
    return np.matmul(e[:,0], pow(data,3)) + np.matmul(e[:,1], pow(data,2)) + np.matmul(e[:,2], pow(data,1)) + np.matmul(e[:,3], np.ones((2,1)))

dataset_creation()