import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dataset_creation():
    Ntrain = [100, 200, 500, 1000, 2000, 5000]
    for Nsamples in Ntrain:
        data, labels = generateData(Nsamples)
#         stacking the data (3 X 100) and labels (1 X 100) to create a data set of size 100 X 4
        data_set = pd.DataFrame(np.transpose(np.vstack((data, labels))))
        plot3(data, labels, 'T')
#         stroing the egenrated data set in a .csv file
        filename = 'training_dataset' + str(Nsamples) + '.csv'
        data_set.to_csv(filename, index = False)

#     creating validation data sets
    NValidation = 100000
    data, labels = generateData(NValidation)
    plot3(data, labels, 'V')
    data_set = pd.DataFrame(np.transpose(np.vstack((data, labels))))
    filename = 'validation_dataset' + str(NValidation) +'.csv'
    data_set.to_csv(filename, index = False)

def generateData(N):
    gmmParameters = {}
#     given that priors are uniform so using 1/4 for each
    gmmParameters['priors'] = [0.25, 0.25, 0.25, 0.25] # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[0, 0, 0], [0, 0, 3], [0, 3, 0], [3, 0, 0]])
    gmmParameters['covMatrices'] = np.zeros((4, 3, 3))
    gmmParameters['covMatrices'][0,:,:] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][1,:,:] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][2,:,:] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][3,:,:] = np.array([[8, 0, 0], [0, 1, 0], [3, 0, 15]])
    x, labels = generateDataFromGMM(N,gmmParameters)
    return x, labels

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[1] # Data dimensionality
    C = len(priors) # Number of components
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        Nl = len(indl[1])
        labels[indl] = (l)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[l,:], covMatrices[l,:,:], Nl))
    labels = np.squeeze(labels)    
    return x,labels

def plot3(data, labels, dtype):
#   from matplotlib import pyplot
#   import pylab
#   from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0,labels == 0], data[1,labels == 0], data[2,labels == 0], marker='o', color='b')
    ax.scatter(data[0,labels == 1], data[1,labels == 1], data[2,labels == 1], marker='^', color='r')
    ax.scatter(data[0,labels == 2], data[1,labels == 2], data[2,labels == 2], marker='*', color='y')
    ax.scatter(data[0,labels == 3], data[1,labels == 3], data[2,labels == 3], marker='+', color='g')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    if (dtype == 'T'):
        ax.set_title('Training Dataset')
    else:
        ax.set_title('Validation Dataset')
    plt.show()

def predict(data, e):
    return np.matmul(e[:,0], pow(data,3)) + np.matmul(e[:,1], pow(data,2)) + np.matmul(e[:,2], pow(data,1)) + np.matmul(e[:,3], np.ones((2,1)))

dataset_creation()