from Observation.observation import Observation
from TrainingsDataManager.imageDataManager import ImageDataManager
from UI.gui import Gui
from CNN.customModel import CustomModel
from Observation.observation import Observation


class Controller:
    """Main controller for the CNN demonstrator."""

    def __init__(self, model: CustomModel, obs: Observation, manager: ImageDataManager):
        """Initializes the controller."""
        self.model = model
        self.observation = obs
        self.manager = manager

        # Setup steps
        self.model.load()
        self.observation.start_webcam()

    def __del__(self):
        """Automatically disconnect the webcam when the controller is
        destroyed."""
        self.stop_observation()

    def stop_observation(self):
        self.observation.stop()

    def predict(self):
        """Takes an image with the camera, predicts the type of coffee
        and returns the image and result to the GUI."""
        frame = self.observation.read_frame()
        results = self.model.predict(frame)

        if len(results) > 0:
            result = results[0]
        else:
            result = "Unknown"
        return frame, result

    def train(self, directory):
        X, y = self.manager.createTrainingData(directory)

        self.model.train(X, y)
        self.model.save()
