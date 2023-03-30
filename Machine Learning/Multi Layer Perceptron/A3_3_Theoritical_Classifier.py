import random
import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)

priors = np.array([[0.25, 0.25, 0.25, 0.25]]) # priors should be a row vector
meanVectors = np.array([[0, 0, 0], [0, 0, 3], [0, 3, 0], [3, 0, 0]])
covMatrices = np.zeros((4, 3, 3))
covMatrices[0,:,:] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
covMatrices[1,:,:] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
covMatrices[2,:,:] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
covMatrices[3,:,:] = np.array([[8, 0, 0], [0, 1, 0], [3, 0, 15]])

# compute class condtional pdf
def compute_class_conditional_pdf(labels, no_labels, no_samples):
    P_x_given_L = np.zeros(shape = [no_labels, no_samples])
    unq_ls = len(np.unique(labels))
    for i in range(unq_ls):
        P_x_given_L[i, :] = multivariate_normal.pdf(input_features, meanVectors[i, :], covMatrices[i, :, :])
    return P_x_given_L

# compute Confusion Matrix
def compute_confusion_matrix (No_labels, class_labels):
    cm = np.zeros(shape = [No_labels, No_labels])
    for i in range(No_labels):
        for j in range(No_labels):
            if j in class_labels and i in class_labels:
                cm[i, j] = (np.size(np.where((i == Decision) & (j == class_labels)))) / np.size(np.where(class_labels == j))
    return cm

# Load your dataset into a Pandas dataframe or NumPy array.
data = pd.read_csv('validation_dataset100000.csv')
# Input features are x and class labels are y 
input_features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
no_labels = len(np.unique(labels))
no_samples = input_features.shape[0]

P_x_given_L = compute_class_conditional_pdf(labels, no_labels, no_samples)
# Compute Class Posteriors using priors and class conditional PDF
P_x = np.matmul(priors, P_x_given_L)
class_posteriors = (P_x_given_L * (np.matlib.repmat(np.transpose(priors), 1, no_samples))) / np.matlib.repmat(P_x, no_labels, 1)

# Define 0-1 Loss Matrix
loss_matrix = np.ones(shape = [no_labels, no_labels]) - np.eye(no_labels)

# Evaluate Expected risk and decisions based on minimum risk
expected_risk = np.matmul(loss_matrix, class_posteriors)
Decision = np.argmin(expected_risk, axis = 0)
avg_exp_risk = np.sum(np.min(expected_risk, axis = 0)) / no_samples
print(f'Average Expected: {avg_exp_risk}')
confusion_matrix = compute_confusion_matrix (no_labels, labels)
print("Confusion Matrix:")
print(confusion_matrix)