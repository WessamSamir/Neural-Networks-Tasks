from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from Implementation import Task1

def open_task(root, screen):
    root.destroy()
    screen()

class Main():
    def __init__(self) :
        
        root=tk.Tk()
        root.title("Neural Networks")
        root.geometry("1125x575")

        image = Image.open("Photos/Background.png")
        background_image = ImageTk.PhotoImage(image)
        background_label = Label(root, image=background_image)

        task1_button_image = PhotoImage(file="Photos/Task 1.png")
        task1_button = Button(root, image=task1_button_image, borderwidth=0, cursor="hand2", bd=0,
                              background='#053654', activebackground='#053654',command=lambda: open_task(root, Task1))




        background_label.place(x=0, y=0)
        task1_button.place(anchor='center', relx=0.80, y=125)

        root.mainloop() 
Main()        