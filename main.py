from CNN.customModel import CustomModel
from Observation.observation import Observation
from Controller.controller import Controller
from UI.gui import Gui
from TrainingsDataManager.imageDataManager import ImageDataManager

if __name__ == '__main__':
    model_name = 'model_name'                                           # TODO
    categories = ["Caffe Americano", "Cappuccino", "Latte Macchiato"]   # TODO
    epochs = 10                                                         # TODO
    image_width = 800                                                   # TODO
    image_height = 600                                                  # TODO

    model = CustomModel(model_name, categories, epochs)
    observation = Observation(image_width, image_height)
    manager = ImageDataManager(image_width, image_height)
    controller = Controller(model, observation, manager)
    gui = Gui(controller)

    gui.root.mainloop()
