import numpy as np
import matplotlib.pyplot as plt
# x represents number of training samples
x = [100, 200, 500, 1000, 2000, 5000]
# elements in error correspond to loss produced by ML model with training samples in x
error = [25.79, 25.49, 23.36, 23.17, 21.71, 25.09]
# Plot graph to compare theoretical classifier and various MLP models
plt.semilogx(x, error, marker = "*", markersize = 18)
plt.axhline(y = 17.903, color = 'b', linestyle = '-')
plt.xlabel("Number of Samples (Semilog axis)")
plt.ylabel("Emperically Estimated P Error")
plt.title("Plot to Compare Errors of the trained models")
plt.show()