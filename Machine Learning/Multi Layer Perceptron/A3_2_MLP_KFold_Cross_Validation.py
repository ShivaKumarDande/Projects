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

# Define the MLP structure with two layers, a hidden layer, and an output layer. 
# The number of perceptrons in the hidden layer should be determined by cross-validation. 
# You can experiment with different smooth-ramp activation functions such as 
# ISRU, Smooth-ReLU, ELU, etc., to see which one works best for your problem.
class MLP(nn.Module):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activationMethod = nn.ELU(alpha = 1.0)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.softmaxRegression = nn.Softmax(dim = 1)
    
    def forward(self, x):
        x = self.layer1(x.float())
        x = self.activationMethod(x)
        x = self.layer2(x)
        x = self.softmaxRegression(x)
        return x

# Train your MLP using the training set. 
# During training, you will need to feed the input data forward 
# through the MLP to compute the output, then backpropagate the 
# error to update the weights. Repeat this process until the MLP converges,
# or until a predetermined stopping criterion is met.
def train(model, loss_fn, optimizer, train_loader):
    # Set the model to train mode
    model.train()
    # Iterate over the batches of training data
    for batch in train_loader:
        # Zero the gradients for this batch
        optimizer.zero_grad()
        # Compute the model's predictions for this batch
        inputs, labels = batch
        outputs = model(inputs.float())
        # Convert the labels tensor to Long type
        labels = labels.long()
        # Compute the loss between the predictions and the ground-truth labels
        loss = loss_fn(outputs, labels)
        # Backpropagate the loss and update the weights
        loss.backward()
        optimizer.step()
    
    return outputs, loss

# Evaluate the model's performance on the validation set after each epoch
def evaluate(model, loss_fn, test_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs.float())
            # Convert the labels tensor to Long type
            labels = labels.long()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += inputs.size(0)
        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
    return val_loss, val_acc, outputs

# Randomly initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def plotBarGraph(average_loss):
    # Plot a bar graph to decide optimal number of perceptrons
    plt.bar(np.linspace(1,10,10), average_loss)
    plt.xlabel("Number of Perceptrons")
    plt.ylabel("Average Loss after 10-Fold cross validation")
    plt.title("Plot of Error vs No of Perceptrons in the Hidden Layer")
    plt.show()

# Load your dataset into a Pandas dataframe or NumPy array.
data = pd.read_csv('training_dataset5000.csv')
input_features = data.iloc[:, : -1].to_numpy()
labels = data.iloc[:, -1].to_numpy()

#X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, : -1].to_numpy(), data.iloc[:, -1].to_numpy(), test_size=0.2, random_state=42)
batch_size = 32
num_folds = 10
kf = KFold(n_splits=num_folds)

minValidationLoss = float("inf") # Assining a large value to minValidationLoss (+ infinity)
maxValidationAccuracy = float("-inf") # Assining a large value to maxValidationAccuracy (- infinity)
optimal_no_of_Perceptrons = 0
average_loss = []

for perceptrons in range(1, 11):
    fold_loss = []
    fold_accuracy = []
    print('---------------------------------------------------------------------')
    print(f"Number of Neurons / Perceptrons in the Hidden Layer : {perceptrons} ")
    print('---------------------------------------------------------------------')
    model = MLP(input_features.shape[1], perceptrons, len(np.unique(labels)))
    model.apply(init_weights)

    # Define the loss function, which in this case, is cross-entropy loss. 
    # The loss function will be used to evaluate how well your MLP is performing.
    import torch.nn.functional as F
    loss_fn = F.cross_entropy

    # Define the optimizer, which will be used to update the weights of your MLP. 
    # The most commonly used optimizer is stochastic gradient descent (SGD), 
    # which updates the weights after each batch of training examples.
    import torch.optim as optim
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)

    for fold, (train_index, test_index) in enumerate(kf.split(input_features)):
        # print(f'Fold {fold + 1}')
        # Split data into train and test sets for the current fold
        X_train, X_test = input_features[train_index], input_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Create PyTorch data loaders for the train and test sets
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(50):
            # Forward pass
            outputs, loss = train(model, loss_fn, optimizer, train_loader)

            # Evaluate your MLP using the testing set. 
            # This will give you an idea of how well your MLP generalizes to new data.
            validationLoss, validationAccuracy, predictions = evaluate(model, loss_fn, test_loader)

            # Print the epoch number, training loss, and validation loss and accuracy
            # if epoch % 200 == 0:
        # print(f'Epoch {epoch}: TRAINING LOSS = {loss:.4f}, VALIDATION LOSS = {validationLoss:.4f}, VALIDATION ACCURACY = {validationAccuracy:.4f}')

        # Evaluate the model's performance on the current fold and save the error
        loss, accuracy, predictions  = evaluate(model, loss_fn, test_loader)
        fold_loss.append(loss)
        fold_accuracy.append(accuracy)

     # Calculate the mean error across all folds for the current number of perceptrons
    mean_error = np.mean(fold_loss)
    mean_accuracy = np.mean(fold_accuracy)
    print(f'Mean error for {perceptrons} perceptrons: {mean_error:.4f}')
    average_loss.append(mean_error)
    
    if (minValidationLoss > mean_error):
        minValidationLoss = min(mean_error, minValidationLoss)
        maxValidationAccuracy = max(mean_accuracy, maxValidationAccuracy)
        optimal_no_of_Perceptrons = perceptrons 

print(f'Optimal number of Perceptrons = {optimal_no_of_Perceptrons} : VALIDATION LOSS = {minValidationLoss:.4f}, VALIDATION ACCURACY = {maxValidationAccuracy:.4f}')

# Finally, visualize the results using Matplotlib or other visualization tools 
# to better understand the performance of your MLP.
plotBarGraph(average_loss)

# Experiment with different parameters, 
# such as the number of perceptrons in the hidden layer, 
# the learning rate of the optimizer, the number of training epochs, etc., 
# to see how they affect the performance of your MLP.

# Load your dataset into a Pandas dataframe or NumPy array.
data = pd.read_csv('training_dataset500.csv')
X_train = data.iloc[:, :-1].to_numpy()
y_train = data.iloc[:, -1].to_numpy()

# Load your dataset into a Pandas dataframe or NumPy array.
data = pd.read_csv('validation_dataset100000.csv')
X_test = data.iloc[:, :-1].to_numpy()
y_test = data.iloc[:, -1].to_numpy()

model = MLP(input_features.shape[1], optimal_no_of_Perceptrons, len(np.unique(labels)))
model.apply(init_weights)
loss_fn = F.cross_entropy
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate,  momentum = 0.9)

# Create PyTorch data loaders for the train and test sets
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, shuffle=True)
batchSize = X_test.shape[0]
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_dataset, batch_size = batchSize, shuffle=False)

for epoch in range(250):
    print(epoch)
    # Forward pass
    outputs, loss = train(model, loss_fn, optimizer, train_loader)

    # Evaluate your MLP using the testing set. 
    # This will give you an idea of how well your MLP generalizes to new data.
    #validationLoss, validationAccuracy, predictions = evaluate(model, loss_fn, test_loader)

y_pred = torch.zeros(0, dtype=torch.long, device='cpu')
# Evaluate the model's performance on the current fold and save the error
loss, accuracy, predictions = evaluate(model, loss_fn, test_loader)
print(f"FINAL LOSS IS {loss}, ACCURACY IS {accuracy}")
y_pred = torch.cat([y_pred, predictions.argmax(1)])
ConfusionMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(ConfusionMatrix)
error = (1 - (ConfusionMatrix[0][0] + ConfusionMatrix[1][1] + ConfusionMatrix[2][2] + ConfusionMatrix[3][3]) / X_test.shape[0])
print("Error:")
print(error)