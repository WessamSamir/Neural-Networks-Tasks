import numpy as np
import pandas as pd
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk






class Task1:
    def __init__(self):

        self.mainColor='#053654'
        self.secondColor = '#053654'
        self.foregroundColor = '#ffffff'
        self.root=tk.Tk()
        self.root.title("Task1")
        self.root.geometry("880x570")

        self.setting_background()

        self.objects()

        self.placing_widgets()

        self.root.mainloop()

    def setting_background(self):
        self.image = Image.open("Photos/Internal_Background.png")
        self.Internal_BG = ImageTk.PhotoImage(self.image)
        self.Internal_BG_label = Label(self.root, image=self.Internal_BG)

    def objects(self):
        self.select_two_feature = Image.open("Photos/Select Two Features.png")
        self.select_two_feature_image = ImageTk.PhotoImage(self.select_two_feature)
        self.select_two_feature_label = Label(self.root, image=self.select_two_feature_image, background=self.mainColor) 



        self.Area_value = StringVar(value=0)
        self.Perimeter_value = StringVar(value=0)
        self.Major_value = StringVar(value=0)
        self.Minor_value = StringVar(value=0)
        self.Roundness_value = StringVar(value=0)

        self.Area_Image = PhotoImage(file="Photos/Area.png")
        self.Area_CheckButton = Checkbutton(self.root, variable=self.Area_value, onvalue="Area",
                                    offvalue="", background=self.mainColor, image=self.Area_Image,
                                    selectimage=self.Area_Image, activebackground=self.mainColor)

        self.Perimeter_Image = PhotoImage(file="Photos/Perimeter.png")
        self.Perimeter_CheckButton = Checkbutton(self.root, variable=self.Perimeter_value, onvalue="Perimeter",
                                         offvalue="", background=self.mainColor, image=self.Perimeter_Image,
                                         selectimage=self.Perimeter_Image, activebackground=self.mainColor)

        self.Major_Image = PhotoImage(file="Photos/Major Axis.png")
        self.Major_CheckButton = Checkbutton(self.root, variable=self.Major_value, onvalue="Major Axis Length",
                                     offvalue="", background=self.mainColor, image=self.Major_Image,
                                     selectimage=self.Major_Image, activebackground=self.mainColor)

        self.Minor_Image = PhotoImage(file="Photos/Minor Axis.png")
        self.Minor_CheckButton = Checkbutton(self.root, variable=self.Minor_value, onvalue="Minor Axis Length",
                                     offvalue="", background=self.mainColor, image=self.Minor_Image,
                                     selectimage=self.Minor_Image, activebackground=self.mainColor)

        self.Roundness_Image = PhotoImage(file="Photos/Roundness.png")
        self.Roundness_CheckButton = Checkbutton(self.root, variable=self.Roundness_value, onvalue="Roundness",
                                         offvalue="", background=self.mainColor, image=self.Roundness_Image,
                                         selectimage=self.Roundness_Image, activebackground=self.mainColor)

        self.select_two_classes = Image.open("Photos/Select Two Classes.png")
        self.select_two_classes_image = ImageTk.PhotoImage(self.select_two_classes)
        self.select_two_classes_label = Label(self.root, image=self.select_two_classes_image, background=self.mainColor)

        self.BOMBAY_value = StringVar(value=0)
        self.CALI_value = StringVar(value=0)
        self.SIRA_value = StringVar(value=0)
        
        self.Bombay_Image = PhotoImage(file="Photos/BOMBAY.png")
        self.Bombay_CheckButton = Checkbutton(self.root, variable=self.BOMBAY_value, onvalue="BOMBAY",
                                     offvalue="", background=self.mainColor, image=self.Bombay_Image,
                                     selectimage=self.Bombay_Image, activebackground=self.mainColor)

        self.Cali_Image = PhotoImage(file="Photos/CALI.png")
        self.Cali_CheckButton = Checkbutton(self.root, variable=self.CALI_value, onvalue="CALI",
                                     offvalue="", background=self.mainColor, image=self.Cali_Image,
                                     selectimage=self.Cali_Image, activebackground=self.mainColor)

        self.Sira_Image = PhotoImage(file="Photos/SIRA.png")
        self.Sira_CheckButton = Checkbutton(self.root, variable=self.SIRA_value, onvalue="SIRA",
                                         offvalue="", background=self.mainColor, image=self.Sira_Image,
                                         selectimage=self.Sira_Image, activebackground=self.mainColor)
        
        self.Enter_Learning_rate = Image.open("Photos/Enter Learning Rate.png")
        self.Enter_Learning_rate_image = ImageTk.PhotoImage(self.Enter_Learning_rate)
        self.Enter_Learning_rate_label = Label(self.root, image=self.Enter_Learning_rate_image, background=self.mainColor)
        
        self.input = Image.open("Photos/Input.png")
        self.input_image = ImageTk.PhotoImage(self.input)
        self.input_label = Label(self.root, image=self.input_image, background=self.mainColor)
        self.number_value = StringVar(value="")
        self.numberEntry = Entry(self.root, width=15, font=("arial", 28), bd=0, textvariable=self.number_value,background=self.mainColor,foreground=self.foregroundColor )


        self.Enter_Epochs = Image.open("Photos/Enter number of Epochs.png")
        self.Enter_Epochs_image = ImageTk.PhotoImage(self.Enter_Epochs)
        self.Enter_Epochs_label = Label(self.root, image=self.Enter_Epochs_image, background=self.mainColor)
        
        self.input2 = Image.open("Photos/Input.png")
        self.input2_image = ImageTk.PhotoImage(self.input2)
        self.input2_label = Label(self.root, image=self.input2_image, background=self.mainColor)
        self.number2_value = StringVar(value="")
        self.number2Entry = Entry(self.root, width=15, font=("arial", 28), bd=0, textvariable=self.number2_value,background=self.mainColor,foreground=self.foregroundColor )


        self.mse = Image.open("Photos/Enter mse Threshold.png")
        self.mse_image = ImageTk.PhotoImage(self.mse)
        self.mse_label = Label(self.root, image=self.mse_image, background=self.mainColor)
        
        self.input3 = Image.open("Photos/Input.png")
        self.input3_image = ImageTk.PhotoImage(self.input3)
        self.input3_label = Label(self.root, image=self.input3_image, background=self.mainColor)
        self.number3_value = StringVar(value="")
        self.number3Entry = Entry(self.root, width=15, font=("arial", 28), bd=0, textvariable=self.number3_value,background=self.mainColor,foreground=self.foregroundColor )
         
        self.type_value = StringVar(value="none")
        self.Bias_Image = PhotoImage(file="Photos/Bias.png")
        self.Bias_radio_Button = Radiobutton(self.root, variable=self.type_value,
                                           value="bias", background=self.mainColor, image=self.Bias_Image,
                                           activebackground=self.mainColor)

        bias=self.type_value.get()

        self.Choose_Algorithm = Image.open("Photos/Choose Algorithm.png")
        self.Choose_Algorithm_image = ImageTk.PhotoImage(self.Choose_Algorithm)
        self.Choose_Algorithm_label = Label(self.root, image=self.Choose_Algorithm_image, background=self.mainColor)

        self.type_value = StringVar(value="none")
        self.Perceptron_Image = PhotoImage(file="Photos/Perceptron.png")
        self.Perceptron_RadioButton = Radiobutton(self.root, variable=self.type_value,
                                           value="bits", background=self.mainColor, image=self.Perceptron_Image,
                                           activebackground=self.mainColor)
        self.Adaline_Image = PhotoImage(file="Photos/Adaline.png")
        self.Adaline_RadioButton = Radiobutton(self.root, variable=self.type_value,
                                             value="level", background=self.mainColor, image=self.Adaline_Image,
                                             activebackground=self.mainColor)

    def placing_widgets(self):
        self.Internal_BG_label.place(x=0, y=0)
        self.select_two_feature_label.place(anchor='center', relx=0.2, y=30)
        self.Area_CheckButton.place(relx=0.05, rely=0.1)
        self.Perimeter_CheckButton.place(relx=0.25, rely=0.1)
        self.Major_CheckButton.place(relx=0.45, rely=0.1)
        self.Minor_CheckButton.place(relx=0.65, rely=0.1)
        self.Roundness_CheckButton.place(relx=0.85, rely=0.1)
        self.select_two_classes_label.place(anchor='center',relx=0.19,y=120)
        self.Bombay_CheckButton.place(relx=0.05,rely=0.25)
        self.Cali_CheckButton.place(relx=0.25,rely=0.25)
        self.Sira_CheckButton.place(relx=0.45,rely=0.25)
        self.Enter_Learning_rate_label.place(anchor='center',relx=0.21,y=210)
        self.input_label.place(anchor='center', relx=0.70, y=210)
        self.numberEntry.place(anchor="center",relx=0.70,y=210)
        self.Enter_Epochs_label.place(anchor='center',relx=0.25,y=290)
        self.input2_label.place(anchor='center', relx=0.70, y=290)
        self.number2Entry.place(anchor="center",relx=0.70,y=290)
        self.mse_label.place(anchor='center',relx=0.21,y=370)
        self.input3_label.place(anchor='center', relx=0.70, y=370)
        self.number3Entry.place(anchor="center",relx=0.70,y=370)
        self.Bias_radio_Button.place(anchor='center', relx=0.06, y=440)
        self.Choose_Algorithm_label.place(anchor='center',relx=0.19,y=500)
        self.Perceptron_RadioButton.place(anchor='center', relx=0.50, y=500)
        self.Adaline_RadioButton.place(anchor='center', relx=0.75, y=500)