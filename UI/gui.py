import random
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import cv2
import os

from Controller.controller import Controller


class Gui:

    observed_image: PhotoImage

    trainings_path = ""

    # TODO create possibility to select your own image via GUI
    image_path = "C:\\Datasets\\PetImages\\Cat\\12.jpg"

    def __init__(self, controller):
        self.controller = controller
        # SETUP GUI
        self.root = Tk()
        # self.root.geometry("500x700")
        self.root.iconbitmap("C:\\Datasets\\coffee_icon.ico")
        self.root.title("Coffee AI")

        # set IMAGE
        if (self.trainings_path == ""):
            self.show_image = Canvas(self.root, width=400, height=400)
        else:
            self.found_image = ImageTk.PhotoImage(Image.open(self.image_path))
            self.show_image = Label(self.root, image=self.found_image, width=400, height=400)

        # set LABELS
        self.results_label = "Results: "
        self.label_text = StringVar()
        self.label_text.set(self.results_label)
        self.label = Label(self.root, textvariable=self.label_text)

        self.output_result_label = StringVar()
        self.output_result_label.set("No results yet...")
        self.output_label = Label(self.root, textvariable=self.output_result_label)

        self.root.filename = StringVar()
        self.root.filename.set("no path selected yet..")
        self.path_label = Label(self.root, textvariable=self.root.filename)

        # set BUTTONS
        self.path_button = Button(self.root, text="Select dataset", command=self.open_directory)
        self.train_button = Button(self.root, text="Train", state=DISABLED)
        self.close_button = Button(self.root, text="Close", command=self.root.quit)
        self.start_button = Button(self.root, text="Start Observer", command=self.start_observer)

        # layout
        self.show_image.grid(row=0, column=0, columnspan=2, rowspan=2, pady=10, padx=10, sticky=W + E + N + S)
        self.label.grid(row=3, column=0, sticky=N, pady=(0, 20))
        self.output_label.grid(row=3, column=1, sticky=N, pady=(0, 20))
        self.path_button.grid(row=4, column=0, columnspan=2, sticky=W + E + N + S, padx=10)
        self.path_label.grid(row=5, column=0, columnspan=2, sticky=W + E + N + S, padx=10)
        self.train_button.grid(row=6, column=0, columnspan=2, sticky=W + E + N + S, padx=10, pady=(10, 10))
        self.start_button.grid(row=7, column=0, sticky=W + E + N + S, pady=10, padx=10)
        self.close_button.grid(row=7, column=1, sticky=W + E + N + S, pady=10, padx=10)

    def start_observer(self):
        # TODO connect with controller
        image_array = self.controller.start_observer()
        self.update_gui(image_array)

    def update_gui(self, array):
        print(array)
        self.show_image.grid_forget()
        img = Image.fromarray(array, mode="RGB")
        imgtk = ImageTk.PhotoImage(img)
        self.show_image = Label(self.root, image=imgtk, width=400, height=400)
        self.show_image.grid(row=0, column=0, columnspan=2, rowspan=2, pady=10, padx=10, sticky=W + E + N + S)
        self.show_image.photo = imgtk

        # if self.trainings_path == "":
        #     self.show_image = Canvas(self.root, width=400, height=400)
        # else:
        #     self.show_image.grid_forget()
        #     self.found_image = ImageTk.PhotoImage(Image.open(self.image_path))
        #     self.show_image = Label(self.root, image=self.found_image, width=400, height=400)
        #     self.show_image.grid(row=0, column=0, columnspan=2, rowspan=2, pady=10, padx=10, sticky=W + E + N + S)

    def open_directory(self):
        # TODO codeline needed for selection image via directory
        # self.root.filename.set(filedialog.askopenfilename(initialdir="C:", title="Select A File", filetypes=(("png files", "*.jpg"), ("all files", "*.*"))))
        self.root.filename.set(filedialog.askdirectory())
        self.trainings_path = self.root.filename.get()

        # TODO implement trainbutton
        self.train_button.grid_forget()
        self.train_button = Button(self.root, text="Train")
        self.train_button.grid(row=6, column=0, columnspan=2, sticky=W + E + N + S, padx=10, pady=(10, 10))
        print((self.trainings_path))

    def show_observation_image(self, array):
        print(array)
        img = Image.fromarray(array, mode="RGB")
        self.observed_image = ImageTk.PhotoImage(img)

    def get_training_directory(self):
        return self.trainings_path
