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
from tkinter import messagebox
import tkinter.ttk as ttk
def perceptron_window(selected_features, selected_classes, learning_rate, epochs,bias,sample):
    def show_result_window(accuracy, confusion_matrix):
        result_window = tk.Toplevel()
        result_window.title("Perceptron Results")

        # Create a label to display the accuracy
        accuracy_label = tk.Label(result_window, text=f"Accuracy: {accuracy:.2f}%")
        accuracy_label.pack()

        # Create a table to display the confusion matrix
        tree = ttk.Treeview(result_window, columns=("Metric", "Value"))
        tree.heading("#1", text="Metric")
        tree.heading("#2", text="Value")

        for metric, value in confusion_matrix.items():
            tree.insert("", "end", values=(metric, value))

        tree.pack()

        # Create a button to close the window
        close_button = tk.Button(result_window, text="Close", command=result_window.destroy)
        close_button.pack()

    data=pd.read_csv('Dry_Bean_Dataset.csv')


    data = data.fillna(np.mean(data['MinorAxisLength']))


    def perceptron(X_train, Y_train, eta, epoc, b):
        w = [random.uniform(0, 1) for _ in range(2)]
        w = np.round(w, 3)
        if b==1:
            b=random.uniform(0,1)
        else :
            b=0
        
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
    

    def classify_single_sample(X_sample, w, b):
       with open('ScalerModel.pkl', 'rb') as file:
        saved_scaler = pickle.load(file)

        scaled_sample = saved_scaler.transform([X_sample])

        f = np.sign(np.dot(scaled_sample, w) + b)
        if f == 1:
            result = selected_classes[0]
        else:
            result = selected_classes[1]
        return result

    def show_classification_result( result):
        # Create a new window to display the classification result
        result_window = tk.Toplevel()
        result_window.title("Classification Result")

        # Create a label to display the classification result
        result_label = tk.Label(result_window, text=f"Classified as: {result}", font=("arial", 12))
        result_label.pack()


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

        accuracy = ( TP + TN ) / len(X_test)
        return accuracy * 100,confusion_matrix

    def split(x1 , x2 , c1 , c2,samples=20):
        cla1 = data[data['Class'] == c1].replace(c1, 1)
        cla2 = data[data['Class'] == c2].replace(c2, -1)
        Y = pd.concat([cla1, cla2], axis=0, ignore_index=True)
        column1 = Y[x1]
        column2 = Y[x2]
        X = [(x, y) for x, y in zip(column1, column2)]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
        with open('ScalerModel.pkl', 'wb') as file:
         pickle.dump(scaler, file)

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

    X_train, X_test, Y_train, Y_test = split(selected_features[0], selected_features[1], selected_classes[0], selected_classes[1], samples=20)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    w , b = perceptron(X_train , Y_train , learning_rate , epochs, bias)

    Accuracy,c = test_perceptron(X_test,Y_test,w,b)
    show_result_window(Accuracy,c)
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(Y_train)
    class_names = {
    -1: selected_classes[0],
    1: selected_classes[1],
            }

    # Loop through unique class labels and plot the data with class names
    for cls in unique_classes:
        class_name = class_names.get(cls, f'Unknown Class {cls}')
        plt.scatter(X_train[Y_train == cls][:, 0], X_train[Y_train == cls][:, 1], label=class_name, marker='o')



    if len(unique_classes) == 2:
        x1 = np.linspace(0, 1, 100)
        x2 = -(w[0] * x1 + b) / w[1]
        plt.plot(x1, x2, color='red', label='Decision Boundary')

    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title('Perceptron Decision Boundary and Classes')
    plt.legend()
    plt.grid(True)
    plt.show()
    s=classify_single_sample(sample,w , b)
    show_classification_result(s)