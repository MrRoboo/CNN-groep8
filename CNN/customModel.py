from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from collections import Counter
import numpy as np


class CustomModel:
    def __init__(self, modelName, categories, epochs):
        self.model = Sequential()
        self.name = modelName
        self.categories = categories
        self.epochs = epochs

    def configure(self, X, y):

        categoryCount = len(Counter(y).keys())

        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(Dense(categoryCount, activation="softmax"))

        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer="adam",
                           metrics=['accuracy'],
                           )

    def load(self):
        self.model = models.load_model(self.name)
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer="adam",
                           metrics=['accuracy'],
                           )

    def save(self):
        self.model.save(self.name, overwrite=True)

    def train(self, X, y):
        batchSize = round(len(X) * 0.01)
        self.model.fit(X, y, batch_size=batchSize, epochs=self.epochs, validation_split=0.1)

    def predict(self, X):
        predictions = self.model.predict([X])
        result = []
        for prediction in predictions:

            highestValue = max(prediction)
            array = np.array(prediction)
            highestValueIndex = list(array).index(highestValue)
            category = self.categories[highestValueIndex]
            result.append(category)

        return result
