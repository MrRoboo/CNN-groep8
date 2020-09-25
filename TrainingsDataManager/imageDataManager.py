import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


class ImageDataManager:

    def __init__(self, image_width, image_height):
        self.dataCategories = []
        self.image_width = image_width
        self.image_height = image_height

    def load(self, directory):
        pickleInX = open(directory + ".X.pickle", "rb")
        pickleInY = open(directory + ".y.pickle", "rb")

        return pickle.load(pickleInX), pickle.load(pickleInY)

    def save(self, directory, X, y):
        pickleOutX = open(directory + ".X.pickle", "wb")
        pickle.dump(X, pickleOutX)
        pickleOutX.close()

        pickleOutY = open(directory + ".y.pickle", "wb")
        pickle.dump(y, pickleOutY)
        pickleOutY.close()

    def __getImageData(self, directory):
        trainingData = []

        for root, dirs, files in os.walk(directory, topdown=False):
            for path in dirs:
                fullPathName = os.path.join(root, path)
                trimmedPathName = os.path.relpath(fullPathName, directory)

                category = trimmedPathName.split('-', 1)[-1]

                if category not in self.dataCategories:
                    self.dataCategories.append(category)

                classNumber = self.dataCategories.index(category)
                for img in os.listdir(fullPathName):
                    try:
                        imageArray = cv2.imread(os.path.join(fullPathName, img), cv2.IMREAD_GRAYSCALE)
                        scaledImageArray = cv2.resize(imageArray, (self.image_width, self.image_height))
                        trainingData.append([scaledImageArray, classNumber])

                    except Exception as e:
                        print(str(e))

        # random.shuffle(trainingData)
        return trainingData

    def createValidationData(self, directory):
        trainingData = self.__getImageData(directory)

        X = []
        y = []

        for data, category in trainingData:
            X.append(data)
            y.append(category)

        X, y = self.__normalize(X, y)
        return X

    def createTrainingData(self, directory):
        trainingData = self.__getImageData(directory)

        X = []
        y = []

        for data, category in trainingData:
            X.append(data)
            y.append(category)

        X, y = self.__normalize(X, y)
        return X, y

    def __normalize(self, X, y):
        X = np.array(X).reshape(-1, (self.image_height, self.image_width, 1))
        y = np.array(y)
        return X / 255.0, y




