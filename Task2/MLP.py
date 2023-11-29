import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('Task2/Dry_Bean_Dataset.csv')
data = data.fillna(np.mean(data['MinorAxisLength']))
y=data['Class']
x = data.drop('Class', axis=1)  
scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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
        actual_output=[data.reshape(-1,1)]
        input_layers=[data.reshape(-1,1)]

        for i in range(len(self.weights)):
            net=np.dot(self.weights[i],input_layers[-1])+self.biases
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
            self.biases+=self.eta*error_siganl[i] 
                                        
