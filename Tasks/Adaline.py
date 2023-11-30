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

def adaline_window(selected_features, selected_classes, learning_rate, epochs,mse_threshold,biaas,sample):
    def show_result_window(accuracy, confusion_matrix):
        result_window = tk.Toplevel()
        result_window.title("Adaline Results")

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

    def classify_single_sample(X_sample, w, b):
        with open('ScalerModel.pkl', 'rb') as file:
            saved_scaler = pickle.load(file)

        scaled_sample = saved_scaler.transform([X_sample])

        temp = np.dot(w, scaled_sample.astype('float64').T) + b
        if temp >= 0:
            result=selected_classes[0]
        else:
            result=selected_classes[1]

        return result

    def show_classification_result( result):
        result_window = tk.Toplevel()
        result_window.title("Classification Result")

        result_label = tk.Label(result_window, text=f"Classified as: {result}", font=("arial", 12))
        result_label.pack()


    def Adaline(X_train,Y_train,eta,epoc,Threshold_mse,b):
        
        w = [random.uniform(0, 1) for _ in range(2)]
        w = np.round(w, 3)
        if b==1:
            b=random.uniform(0,1)
        else:
            b=0
        for _ in range(epoc):

          
            errors = []
            for i in range(len(X_train)):
                X = X_train[i]
                t = Y_train[i]

                v = np.dot(w, X) + b
                y = v  
                error = t - y
                errors.append(error)
                b += eta * error
                w += eta * error * X
            mse = np.mean(np.square(errors))

            if mse < Threshold_mse:
                break
        return w, b 


    def test_Adaline(X_test, Y_test, w, b):
        misclassified_samples = 0
        y_pre = []
        for i in range(len(X_test)):
            temp = np.dot(w, X_test[i].astype('float64')) + b
            if temp >= 0:
                y_pre.append(1)
            else:
                y_pre.append(-1)

            if y_pre[i] != Y_test[i]:
                misclassified_samples += 1

        Y_test = np.array(Y_test)
        y_pre = np.array(y_pre)

        TP = np.sum((Y_test == 1) & (y_pre == 1))
        TN = np.sum((Y_test == -1) & (y_pre == -1))
        FP = np.sum((Y_test == -1) & (y_pre == 1))
        FN = np.sum((Y_test == 1) & (y_pre == -1))

        confusion_matrix = {
            'True Positives (TP)': TP,
            'True Negatives (TN)': TN,
            'False Positives (FP)': FP,
            'False Negatives (FN)': FN
        }
        # for key, value in confusion_matrix.items():
        #     print(f'{key}: {value}')

        accuracy = (TP + TN) / len(Y_test)
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

    X_train, X_test, Y_train, Y_test = split(selected_features[0], selected_features[1],selected_classes[0], selected_classes[1], samples=20)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    w , b = Adaline(X_train , Y_train , learning_rate, epochs, mse_threshold,biaas)
    Accuracy,c= test_Adaline(X_test,Y_test,w,b)
    show_result_window(Accuracy, c)


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
    result=classify_single_sample(sample,w,b)
    show_classification_result(result)
        