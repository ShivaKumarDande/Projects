import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from random import seed
from random import randrange
from scipy.stats import multivariate_normal

# Load your dataset into a Pandas dataframe or NumPy array.
data = pd.read_csv('GMM_training_dataset10000.csv')
input_features = data.iloc[:, : -1].to_numpy()
labels = data.iloc[:, -1].to_numpy()

# Define the number of folds for cross-validation
K = 10

# Initialize KFold
kf = KFold(n_splits=K)

# Define the range of Gaussian components to evaluate
num_components = [1, 2, 3, 4, 5, 6]

order = []

# Initialize a list to store the average log-likelihoods for each GMM
avg_log_likelihoods = []

for routine in range(30):
    avg_log_likelihoods = []

    for num in num_components:
        # Initialize a list to store the log-likelihoods for each fold
        log_likelihoods = []
        for train_index, val_index in kf.split(input_features):
            # Split the data into training and validation sets
            X_train, X_val = input_features[train_index], input_features[val_index]
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
    # Identify model order with maximum score
    order.append(np.argmax(avg_log_likelihoods) + 1)
    #print(f'order is : {order}')

#print(order)         
best_num_components = num_components[np.argmax(avg_log_likelihoods)]
# best_gmm = GaussianMixture(n_components=best_num_components, random_state=42)
# best_gmm.fit(input_features)

print(f"Best number of Gaussian components: {best_num_components}")

 #Plot a histogram showing frequency of model orders selected
plt.hist(order, range = (1, 7))
plt.xlabel("Model Order")
plt.ylabel("Avg Log Likelihood")
plt.title("Model Order Selection")
plt.show()