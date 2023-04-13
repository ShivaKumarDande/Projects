# Load the necessary Python libraries 
# such as NumPy, Pandas, Matplotlib, Scikit-learn, and PyTorch.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load your dataset into a Pandas dataframe or NumPy array.
data = pd.read_csv('training_dataset1000.csv')
input_features = data.iloc[:, : -1].values
sc = StandardScaler()
x_train = sc.fit_transform(input_features)
y_train = data.iloc[:, -1].values

# Load your dataset into a Pandas dataframe or NumPy array.
data = pd.read_csv('validation_dataset10000.csv')
input_features = data.iloc[:, : -1].values
x_test = sc.transform(input_features)
y_test = data.iloc[:, -1].values

# Define range of hyperparameters for cross validation
C = [0.01, 0.1, 1, 10, 100, 1000, 10000]
gamma = [0.1, 0.01, 0.001, 0.0001]
hyperParameters = {"kernel": ["rbf"], "gamma": gamma, "C": C}

#X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, : -1].to_numpy(), data.iloc[:, -1].to_numpy(), test_size=0.2, random_state=42)
batch_size = 32
num_folds = 10
kf = KFold(n_splits=num_folds)

# Perform cross validation and identify best results 
clf = GridSearchCV(SVC(), hyperParameters, cv = kf)
clf.fit(x_train, y_train)

# printing the parameters for which we get the best results
print(clf.best_params_)

# Print average accuracy scores for all combinations
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

scores = np.array(means).reshape(7, 4)

plt.figure(figsize=(10, 10))
plt.subplots_adjust(left = 0.15, right = 0.6, bottom = 0.3, top = 0.95)
plt.imshow(scores, interpolation='nearest', cmap = plt.cm.nipy_spectral)
plt.xlabel('gamma')
plt.ylabel('C')
plt.title("Color Map for selecting hyperparameters")
plt.xticks(np.arange(len(gamma)), gamma, rotation=45)
plt.yticks(np.arange(len(C)), C)
plt.colorbar()
plt.show()

# Train final SVM classifier
classifier = SVC(C=10, kernel = 'rbf', gamma = 0.1, random_state = 0)
classifier.fit(x_train, y_train)

# Test the model with appropriate evaulation metrics
predictions = classifier.predict(x_test)
print(predictions)
ConfusionMatrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(ConfusionMatrix)
print("The accuracy of the model is {} %".format(str(round(accuracy_score(y_test,predictions),4)*100)))

# Define range of points for identifying classification boundary
x1_min, x1_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
x2_min, x2_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max), np.arange(x2_min, x2_max))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot classification boundary superimposed over testing data
plt.scatter(x_test[:, 0], x_test[:, 1], c = y_test, cmap = "brg", s = 20, edgecolors = 'y')
plt.contourf(xx, yy, Z, cmap = "brg", alpha = 0.3)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Plot showing decision boundary")
plt.show()