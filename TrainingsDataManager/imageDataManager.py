import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


class ImageDataManager:

    def __init__(self, directory):
        self.directory = directory
        self.dataCategories = []

    def load(self):
        pickleInX = open(self.directory + ".X.pickle", "rb")
        pickleInY = open(self.directory + ".y.pickle", "rb")

        return pickle.load(pickleInX), pickle.load(pickleInY)

    def save(self, X, y):
        pickleOutX = open(self.directory + ".X.pickle", "wb")
        pickle.dump(X, pickleOutX)
        pickleOutX.close()

        pickleOutY = open(self.directory + ".y.pickle", "wb")
        pickle.dump(y, pickleOutY)
        pickleOutY.close()

    def __getImageData(self, directory, imageSize):
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
                        scaledImageArray = cv2.resize(imageArray, (imageSize, imageSize))
                        trainingData.append([scaledImageArray, classNumber])

                    except Exception as e:
                        print(str(e))

        # random.shuffle(trainingData)
        return trainingData

    def createValidationData(self, directory, imageSize):
        trainingData = self.__getImageData(directory, imageSize)

        X = []
        y = []

        for data, category in trainingData:
            X.append(data)
            y.append(category)

        X, y = self.__normalize(X, y, imageSize)
        return X

    def createTrainingData(self, imageSize):
        trainingData = self.__getImageData(self.directory, imageSize)

        X = []
        y = []

        for data, category in trainingData:
            X.append(data)
            y.append(category)

        X, y = self.__normalize(X, y, imageSize)
        return X, y

    def __normalize(self, X, y, imageSize):
        X = np.array(X).reshape(-1, imageSize, imageSize, 1)
        y = np.array(y)
        return X / 255.0, y




