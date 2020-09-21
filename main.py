from TrainingsDataManager.imageDataManager import ImageDataManager
from CNN.customModel import CustomModel

dataManager = ImageDataManager("C:/Test/Training")
X, y = dataManager.createTrainingData(50)

# model = CustomModel("modelDogTest", X, y)
# model.configure()
# model.train(5)
# model.save()

model2 = CustomModel("modelDogTest", X, y)
model2.load()
# model2.train(5)

validationData = dataManager.createValidationData("C:/Test/Validation", 50)
result = model2.predict(validationData, dataManager.dataCategories)

print(result)
