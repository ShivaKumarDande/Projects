# -*- coding: utf-8 -*-
"""Logistic_Regression_titanic_data_set.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aislxHj_EmJpTan1Bl-ThgOpIlPnuoyY
"""

import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.head()

test_data.head()

#checking if any null values are present in the data
train_data.isnull().sum()
#Null values are present in age, Embarked and cabin (ignore cabin as they are many null values)
#Let's fill those empty values

train_data['Age'].mean(skipna=True)

train_data['Age'].median(skipna=True)

#Filling the missing data in age with median of age
train_data["Age"].fillna(train_data["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].value_counts().idxmax(), inplace=True)

#checking if any null values are present in the data
train_data.isnull().sum()

train_data.corr()

max1 = train_data['Fare'].max()
max2 = train_data['Parch'].max()

print (max1, max2)

#Normalising the data Fare and Parch
train_data['Fare'] = train_data['Fare'] / max1
train_data['Parch'] = train_data['Parch'] / max2

test_data['Fare'] = test_data['Fare'] / max1
test_data['Parch'] = test_data['Parch'] / max2

x1 = train_data['Fare']
x2 = train_data['Parch']
y = train_data['Survived']

#We have positive correlation only with parch and Fare
#lets takes x1 = Parch, x2 = Fare and y = Survived
def initialize():
  a = 0
  b = 0
  c = 0
  return a, b, c

def costFunction(prediction, y):
  cost1 = np.multiply(y, np.log(prediction+1e-8)) 
  cost2 = np.multiply((1-y), np.log(1-prediction+1e-8))
  cost = np.sum(cost1 + cost2)
  cost = (-1 * cost) / (2*len(y))
  return cost

def predict(a, b, c, x1, x2):
  prediction = 1 / (1 + np.exp(-(np.dot(a, x1)+np.dot(b, x2)+c)))
  prediction = np.where(prediction <= 0.5, 0, 1)
  return prediction

def gradientCalculation(prediction, x1, x2, y):
  m = len(x1)
  da = np.sum(np.multiply(np.subtract(prediction, y), x1)) / m
  db = np.sum(np.multiply(np.subtract(prediction, y), x2)) / m
  dc = np.sum(np.subtract(prediction, y)) / m
  return da, db, dc

def train(x1, x2, y, alpha, iterations):
  costs = []
  a, b, c = initialize()
  for i in range(iterations):
    prediction = predict(a, b, c, x1, x2)
    cost = costFunction(prediction, y)
    da, db, dc = gradientCalculation(prediction, x1, x2, y)
    a = a - (alpha * da)
    b = b - (alpha * db)
    c = c - (alpha * dc)
    costs.append(cost)

    # Print cost every at intervals 10 times or as many iterations if < 10
    if i% math.ceil(iterations/10) == 0:
      print(f"Iteration {i:4}: Cost {float(cost):8.2f}   ")
  return a, b, c, costs

a, b, c, cost = train(x1, x2, y, 0.5, 50000)
print(f'Estimated parameters are {a}, {b}, {c}')
print(f'{cost[-1]} \n')

test_input1 = 34.5
test_input2 = 0
test_input1 = test_input1 / max1 # preprocess
test_input2 = test_input2 / max2
predict(a, b, c, test_input1, test_input2) # prediction

predict(a, b, c, test_data['Fare'], test_data['Parch'])



