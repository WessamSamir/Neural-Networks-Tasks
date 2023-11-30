import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle


class MLP:
    def __init__(self,input_size,hidden_layer,neurons,output_size,eta,activation_fun):
        self.input_size=input_size
        self.hiddin_layer=hidden_layer
        self.neurons=neurons
        self.output_size=output_size
        self.eta=eta
        self.activation_fun=activation_fun

        self.weights=[]
        self.biases=[]

        ############_____Inintialize weights______########
        no_of_layers=[input_size]+neurons+[output_size]
        for i in range(len(no_of_layers)-1):
            weightss=np.random.rand(no_of_layers[i+1],no_of_layers[i])
            self.weights.append(weightss)
            self.biases.append(np.random.rand(no_of_layers[i+1],1))


    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoid_deriv(self,y):
        return y*(1-y)
    
    def tanh(self,z):
        return np.tanh(z)
    
    def tanh_deriv(self,y):
        return 1-np.tanh(y)**2
    def forward_step(self,data):
        data=np.array(data)
        actual_output=[data.reshape(-1,1)]
        input_layers=[data.reshape(-1,1)]
        for i in range(len(self.weights)):
            net=np.dot(self.weights[i],input_layers[i])+self.biases[i]
            if self.activation_fun=='sigmoid':
                activation=self.sigmoid(net)
            else:
                activation=self.tanh(net)    
            actual_output.append(activation)
            input_layers.append(net)   
        return actual_output,input_layers

    def backward_step(self,desired,actual_output,input_layers):      
        desired=desired.reshape(-1,1)
        error_signal=[None]*len(self.weights)

        # for each output unit :-
        if self.activation_fun=='sigmoid':
            e=desired-actual_output[-1]
            error_signal[-1]=e*self.sigmoid_deriv(actual_output[-1])
        else:
            e=desired-actual_output[-1]
            error_signal[-1]=e*self.tanh_deriv(actual_output[-1])    
        # for each hidden unit:-
        for i in reversed(range(len(error_signal)-1)):
            if self.activation_fun=='sigmoid':
                error_signal[i]=np.dot(self.weights[i+1].T,error_signal[i+1])*self.sigmoid_deriv(actual_output[i+1])
            else:
                error_signal[i]=np.dot(self.weights[i+1].T,error_signal[i+1])*self.tanh_deriv(actual_output[i+1])
        return error_signal

    def update_weights(self,error_siganl,actual_output):
        for i in range(len(self.weights)):
            if i ==0:
                input=actual_output[0].reshape(-1,1)
            else:
                input=actual_output[i].reshape(-1,1)
            self.weights[i]+=self.eta*np.dot(error_siganl[i],input.T)
            #self.biases+=self.eta*error_siganl[i] 
       
def MLPer(hidden_layer,neurons,eta,epochs,activation_fun):                                             
    data=pd.read_csv('Dry_Bean_Dataset.csv')
    data = data.fillna(np.mean(data['MinorAxisLength']))
    y=data['Class']
    x = data.drop('Class', axis=1)  
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    with open('Scaler.pkl', 'wb') as file:
     pickle.dump(scaler, file)
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)
    from sklearn.model_selection import train_test_split
    # Separate data for training and testing
   # Separate data for training and testing
    x_train, x_test, y_train, y_test = [], [], [], []
    classes = np.unique(y)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        x_cls = x[cls_indices]
        y_cls = y[cls_indices]

        # Select the first 30 samples for training
        x_train.append(x_cls[:30])
        y_train.extend(y_cls[:30])

        # Select the remaining 20 samples for testing
        x_test.append(x_cls[30:50])
        y_test.extend(y_cls[30:50])

    # Convert lists to NumPy arrays
    x_train = np.concatenate(x_train)
    x_test = np.concatenate(x_test)

    y_train= np.array(y_train)
    y_test = np.array(y_test)

    input_size = x_train.shape[1]
    neurons = neurons  
    output_size = len(np.unique(y))

    input_size = x_train.shape[1]

    neurons = neurons  
    output_size = len(np.unique(y))

    mlp = MLP(input_size, hidden_layer, neurons, output_size, eta, activation_fun)

    # Training loop
    epochs = 1000
    correct_train_predictions = 0  # Initialize counter for correct training predictions

    for epoch in range(epochs):
        correct_train_predictions = 0
        for i in range(len(x_train)):
            actual_output, input_layers = mlp.forward_step(x_train[i])
            error_signal = mlp.backward_step(np.eye(output_size)[y_train[i]], actual_output, input_layers)
            mlp.update_weights(error_signal, actual_output)

            _, train_output = mlp.forward_step(x_train[i])
            predicted_train_class = np.argmax(train_output[-1])
            if predicted_train_class == y_train[i]:
                correct_train_predictions += 1
    # Step 3: Test the Neural Network
    with open('mlp_model.pkl', 'wb') as model_file:
        pickle.dump(mlp, model_file)

    train_accuracy = correct_train_predictions / len(y_train) 
    print(f"Accuracy on the training set: {train_accuracy}")

    predicted_labels = []
    actual_labels = []

    correct_predictions = 0
    for i in range(len(x_test)):
        _, test_output = mlp.forward_step(x_test[i])
        predicted_class = np.argmax(test_output[-1])
        
        # Store predicted and actual labels for later use in the confusion matrix
        predicted_labels.append(predicted_class)
        actual_labels.append(y_test[i])
        
        if predicted_class == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_test)
    print(f"Accuracy on the test set: {accuracy}")

    confusion_matrix = calculate_confusion_matrix(actual_labels, predicted_labels)

    print(f"True Positives (TP): {confusion_matrix['True Positives (TP)']}")
    print(f"True Negatives (TN): {confusion_matrix['True Negatives (TN)']}")
    print(f"False Positives (FP): {confusion_matrix['False Positives (FP)']}")
    print(f"False Negatives (FN): {confusion_matrix['False Negatives (FN)']}")

    accuracy = (confusion_matrix['True Positives (TP)'] + confusion_matrix['True Negatives (TN)']) / len(y_test)

    return accuracy * 100, confusion_matrix

def predict_sample(x_test):
    with open('Scaler.pkl', 'rb') as file:
        saved_scaler = pickle.load(file)

    scaled_sample = saved_scaler.transform([x_test])
    print(scaled_sample)
    with open('mlp_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    _, test_output = loaded_model.forward_step(scaled_sample)
    predicted_test_class = np.argmax(test_output[-1])
    if(predicted_test_class==0):
        print('BOMBAY')
    elif(predicted_test_class==1):
        print('CALI')
    else:
        print('SIRA')

def calculate_confusion_matrix(actual_labels, predicted_labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == 0 and predicted == 0:
            TN += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        elif actual == 1 and predicted == 0:
            FN += 1

    confusion_matrix = {
        'True Positives (TP)': TP,
        'True Negatives (TN)': TN,
        'False Positives (FP)': FP,
        'False Negatives (FN)': FN
    }

    return confusion_matrix
# Calculate confusion matrix
