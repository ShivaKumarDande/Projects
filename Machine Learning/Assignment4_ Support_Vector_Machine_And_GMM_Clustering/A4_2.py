import numpy as np
import sys
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal
np.set_printoptions(threshold=sys.maxsize)

# Read the image from the specified path
image = cv.imread('101085.jpg')
print(image.shape)

# Downsample the image with specified scale
scaling_percentage = 60 # percent of original size
w = int(image.shape[1] * (scaling_percentage / 100))
h = int(image.shape[0] * (scaling_percentage / 100))
dim = (w, h)

# Resizing the input image
img = cv.resize(image, dim, interpolation = cv.INTER_AREA)

# Intialize parameters of the image
rows = img.shape[0]
columns = img.shape[1]
channels = img.shape[2]
pixels = rows * columns
features = 5

# Create raw feature vector
feature_vector = np.zeros(shape = (pixels, features))
pixel = 0
for row in range(rows):
    for col in range(columns):
        feature_vector[pixel, 0] = row
        feature_vector[pixel, 1] = col
        for channel in range(channels):
            feature_vector[pixel, channel + 2] = img[row, col, channel]
        pixel += 1

# Normalize the feature values
sc = MinMaxScaler()
feature_vector = sc.fit_transform(feature_vector)

# Define number of folds and list to store model orders
kf = KFold(n_splits = 10)
avg_log_likelihoods = []
order =[]
num_components = [1, 2, 3, 4, 5, 6]

# Test model orders ranging from 1 to 6
for num in num_components:
    # Initialize a list to store the log-likelihoods for each fold
    log_likelihoods = []
    for train_index, val_index in kf.split(feature_vector):
        # Split the data into training and validation sets
        X_train, X_val = feature_vector[train_index], feature_vector[val_index]
        # Fit the GMM using the EM algorithm
        gmm = GaussianMixture(n_components=num, init_params = 'random', max_iter = 3000, tol = 1e-5, n_init = 2)
        gmm.fit(X_train)

        # Calculate the log-likelihood of the validation set
        log_likelihood = gmm.score(X_val)

        # Append the log-likelihood to the list for this fold
        log_likelihoods.append(log_likelihood)

    # Calculate the average log-likelihood across all K folds
    avg_log_likelihood = np.mean(log_likelihoods)
    #print(log_likelihoods)
    print(avg_log_likelihood)

    # Append the average log-likelihood to the list for all GMMs
    avg_log_likelihoods.append(avg_log_likelihood)

print(avg_log_likelihoods)
# Plot Log Likelihood Score with respect to number of components in each model
plt.plot(np.linspace(1, 6, 6), avg_log_likelihoods)
plt.xlabel("Number of Components")
plt.ylabel("Average Log Likelihood on Validation Datasets")
plt.title("Plot to Identify Number of Components yielding Maximum Score")
plt.show()

# Final Model Fitting
best_num_components = num_components[np.argmax(avg_log_likelihoods)]
print(best_num_components)
best_gmm_model = GaussianMixture(n_components = best_num_components, init_params='random', max_iter = 3000, tol = 1e-8, n_init = 3)
best_gmm_model.fit(feature_vector)

# Compute class posteriors using model weights and conditional probabilties
posteriors = np.zeros((best_num_components, pixels))
for i in range(best_num_components):
    PDF = multivariate_normal.pdf(feature_vector, mean = best_gmm_model.means_[i,:], cov = best_gmm_model.covariances_[i,:,:])
    posteriors[i, :] = (best_gmm_model.weights_[i] * PDF)

# Decide label for each pixel with maximum posterior value
img_labels = np.argmax(posteriors, axis = 0)

# Plot segmented image
plt.imshow(img_labels.reshape(rows, columns))
plt.show()