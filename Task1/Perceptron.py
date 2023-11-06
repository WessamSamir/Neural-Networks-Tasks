import numpy as np
import pandas as pd
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import random
import pickle
from GUI import Task1




bias="None"



data=pd.read_csv('Dry_Bean_Dataset.csv')
print(data)


data.info()


data=data.fillna(data.mean())
print("Data after filling nulls",data)

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outlier_threshold = 1.5
outliers = ((data < (Q1 - outlier_threshold * IQR)) | (data > (Q3 + outlier_threshold * IQR)))

print(data)

plt.figure(figsize=(12, 8))
plt.scatter(
    data.index,
    np.sum(outliers, axis=1),
    c='r',
    marker='o',
    label='Outliers'
)
plt.xlabel('Data Point Index')
plt.ylabel('Number of Outliers')
plt.title('Outliers in the Dataset')
plt.legend()
plt.show()

data[outliers] = np.nan
data = data.fillna(data.mean())

print(data)


def perceptron(X_train, Y_train, eta, epoc, b=0):
    # num_features = X_train.shape[1]
    # w = np.random.rand(num_features)
    w = [random.uniform(0, 1) for _ in range(2)]
    w = np.round(w, 3)
    if bias=='bias':
        b=random.uniform(0,1)
    check = 0
    y_pre = []
    while check < epoc:
        converged = True
        for i in range(len(X_train)):
            f = np.dot(w, X_train[i]) + b
            y_pre.append(np.sign(f))
            if y_pre[i] != Y_train[i]:
                converged = False
                loss = Y_train[i] - y_pre[i]
                w += np.array(eta * loss * X_train[i])
                b = b + eta * loss
        if converged:
            break
        check += 1
    return w, b




def test_perceptron(X_test, Y_test, w, b):
    misclassified_samples = 0
    y_pre = []
    for i in range(len(X_test)):
        temp = np.dot(w, X_test[i]) + b
        y_pre.append(np.sign(temp))
        if y_pre[i] != Y_test[i]:
            misclassified_samples += 1

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for actual, predicted in zip(Y_test, y_pre):
        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == -1 and predicted == -1:
            TN += 1
        elif actual == -1 and predicted == 1:
            FP += 1
        elif actual == 1 and predicted == -1:
            FN += 1

    
    confusion_matrix = {
        'True Positives (TP)': TP,
        'True Negatives (TN)': TN,
        'False Positives (FP)': FP,
        'False Negatives (FN)': FN
    }
    for key, value in confusion_matrix.items():
        print(f'{key}:{value}')

    accuracy = ( TP + TN ) / len(X_test)
    return accuracy * 100

def split(x1 , x2 , c1 , c2,samples=20):
    cla1 = data[data['Class'] == c1].replace(c1, 1)
    cla2 = data[data['Class'] == c2].replace(c2, -1)
    Y = pd.concat([cla1, cla2], axis=0, ignore_index=True)
    column1 = Y[x1]
    column2 = Y[x2]
    X = [(x, y) for x, y in zip(column1, column2)]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    class1_indices = np.where(Y['Class'] == 1)[0]
    class2_indices = np.where(Y['Class'] == -1)[0]

    random.seed(1)
    selected_indices = (random.sample(list(class1_indices), samples) +
                        random.sample(list(class2_indices), samples))

    X_test = X[selected_indices]
    Y_test = Y['Class'][selected_indices]

    remaining_indices = np.setdiff1d(np.arange(len(X)), selected_indices)
    X_train = X[remaining_indices]
    Y_train = Y['Class'][remaining_indices]

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = split('Area', 'roundnes', 'SIRA', 'CALI', samples=20)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

#w , b = perceptron(X_train , Y_train , 0.5 , 100, 0.001)
# print(w)
# with open('Perceptron.pkl', 'wb') as file:
#     pickle.dump(perceptron(X_train , Y_train , 0.5 , 100, 0.001), file)

with open('Perceptron.pkl', 'rb') as file:
    w , b = pickle.load(file)
Accuracy = test_perceptron(X_test,Y_test,w,b)
print(Accuracy)

plt.figure(figsize=(8, 6))
unique_classes = np.unique(Y_train)
for cls in unique_classes:
    plt.scatter(X_train[Y_train == cls][:, 0], X_train[Y_train == cls][:, 1], label=f'Class {cls}', marker='o')


if len(unique_classes) == 2:
    x1 = np.linspace(0, 1, 100)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, color='red', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary and Classes')
plt.legend()
plt.grid(True)
plt.show()















        
        

        
        

