import numpy as np
import pandas as pd
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk


class Task2:
     def __init__(self):

        self.mainColor='#053654'
        self.secondColor = '#053654'
        self.foregroundColor = '#ffffff'
        self.root=tk.Tk()
        self.root.title("Task2")
        self.root.geometry("885x583")

        self.setting_background()
        self.objects()
        self.placing_widgets()
        self.root.mainloop()

     def setting_background(self):
        self.image = Image.open("Photos/Internal_Background.png")
        self.Internal_BG = ImageTk.PhotoImage(self.image)
        self.Internal_BG_label = Label(self.root, image=self.Internal_BG) 


     def objects(self):
        self.hidden_layer = Image.open("Photos/Hidden layers.png")
        self.hidden_layer_image = ImageTk.PhotoImage(self.hidden_layer)
        self.hidden_layer_label = Label(self.root, image=self.hidden_layer_image, background=self.mainColor)     
        
        self.input = Image.open("Photos/Input.png")
        self.input_image = ImageTk.PhotoImage(self.input)
        self.input_label = Label(self.root, image=self.input_image, background=self.mainColor)
        self.number_value = StringVar(value="")
        self.numberEntry = Entry(self.root, width=15, font=("arial", 28), bd=0, textvariable=self.number_value,background=self.mainColor,foreground=self.foregroundColor ) 


        self.NoOfNeuron = Image.open("Photos/No of neurons.png")
        self.NoOfNeuron_image = ImageTk.PhotoImage(self.NoOfNeuron)
        self.NoOfNeuron_label = Label(self.root, image=self.NoOfNeuron_image, background=self.mainColor)

        self.input2 = Image.open("Photos/Input.png")
        self.input_2_image = ImageTk.PhotoImage(self.input2)
        self.input_2_label = Label(self.root, image=self.input_2_image, background=self.mainColor)
        self.number_2_value = StringVar(value="")
        self.numberEntry2 = Entry(self.root, width=15, font=("arial", 28), bd=0, textvariable=self.number_2_value,background=self.mainColor,foreground=self.foregroundColor )


        self.Enter_Learning_rate = Image.open("Photos/Learning Rate.png")
        self.Enter_Learning_rate_image = ImageTk.PhotoImage(self.Enter_Learning_rate)
        self.Enter_Learning_rate_label = Label(self.root, image=self.Enter_Learning_rate_image, background=self.mainColor)

        self.input3 = Image.open("Photos/Input.png")
        self.input_3_image = ImageTk.PhotoImage(self.input3)
        self.input_3_label = Label(self.root, image=self.input_3_image, background=self.mainColor)
        self.number_3_value = StringVar(value="")
        self.numberEntry3 = Entry(self.root, width=15, font=("arial", 28), bd=0, textvariable=self.number_3_value,background=self.mainColor,foreground=self.foregroundColor )

        self.Enter_Epochs = Image.open("Photos/No of Epochs.png")
        self.Enter_Epochs_image = ImageTk.PhotoImage(self.Enter_Epochs)
        self.Enter_Epochs_label = Label(self.root, image=self.Enter_Epochs_image, background=self.mainColor)
        
        self.input4 = Image.open("Photos/Input.png")
        self.input4_image = ImageTk.PhotoImage(self.input4)
        self.input4_label = Label(self.root, image=self.input4_image, background=self.mainColor)
        self.number4_value = StringVar(value="")
        self.number4Entry = Entry(self.root, width=15, font=("arial", 28), bd=0, textvariable=self.number4_value,background=self.mainColor,foreground=self.foregroundColor )

        self.bias_value = StringVar(value=0)
        self.Bias_Image = PhotoImage(file="Photos/Bias.png")
        self.Bias_radio_Button = Checkbutton(self.root, variable=self.bias_value,
                onvalue=1,offvalue=0, background=self.mainColor, image=self.Bias_Image, activebackground=self.mainColor)
        
        self.Choose_Activation = Image.open("Photos/Activation Function.png")
        self.Choose_Activation_image = ImageTk.PhotoImage(self.Choose_Activation)
        self.Choose_Activation_label = Label(self.root, image=self.Choose_Activation_image, background=self.mainColor)

        self.type_value = StringVar(value="none")
        self.Sigmoid_Image = PhotoImage(file="Photos/Sigmoid.png")
        self.Sigmoid_RadioButton = Radiobutton(self.root, variable=self.type_value,
                                           value="bits", background=self.mainColor, image=self.Sigmoid_Image,
                                           activebackground=self.mainColor)
        self.Tangent_Image = PhotoImage(file="Photos/Tangent.png")
        self.Tangent_RadioButton = Radiobutton(self.root, variable=self.type_value,
                                             value="level", background=self.mainColor, image=self.Tangent_Image,
                                             activebackground=self.mainColor)
        
        self.submit_button_image = PhotoImage(file="Photos/Submit.png")
        self.submit_button = Button(self.root, image=self.submit_button_image, borderwidth=0, cursor="hand2", bd=0,
                              background=self.mainColor, activebackground=self.mainColor)

     def placing_widgets(self):
        self.Internal_BG_label.place(x=0, y=0)
        self.hidden_layer_label.place(anchor='center', relx=0.15, y=40)  
        self.input_label.place(anchor='center', relx=0.70, y=50)
        self.numberEntry.place(anchor="center",relx=0.70,y=50)

        self.NoOfNeuron_label.place(anchor='center',relx=0.15,y=110)
        self.input_2_label.place(anchor='center', relx=0.70, y=125)
        self.numberEntry2.place(anchor="center",relx=0.70,y=125)

        self.Enter_Learning_rate_label.place(anchor='center',relx=0.15,y=185)
        self.input_3_label.place(anchor='center', relx=0.70, y=200)
        self.numberEntry3.place(anchor="center",relx=0.70,y=200)

        self.Enter_Epochs_label.place(anchor='center',relx=0.14,y=260)
        self.input4_label.place(anchor='center', relx=0.70, y=275)
        self.number4Entry.place(anchor="center",relx=0.70,y=275)

        self.Bias_radio_Button.place(anchor='center', relx=0.07, y=340)

        self.Choose_Activation_label.place(anchor='center',relx=0.21,y=420)
        self.Sigmoid_RadioButton.place(anchor='center', relx=0.60, y=420)
        self.Tangent_RadioButton.place(anchor='center', relx=0.85, y=420)

        self.submit_button.place(anchor='center', relx=0.50, y=510)

             