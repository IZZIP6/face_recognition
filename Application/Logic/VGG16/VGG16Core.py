from Application.Logic.VGG16.VGG16Model.VGG16 import VGG16
import keras
from keras import Model
from keras_preprocessing import image
import numpy as np
from numpy.linalg import norm


class VGG16Core:
    def __init__(self):
        self.image_path = ""
        keras.backend.set_image_dim_ordering("tf")
        self.original_model = VGG16().VGG16Model
        self.extract_feature_model = Model(input=self.original_model.input,
                                           output=self.original_model.get_layer('fc7').output)

    def make_prediction(self, img2be_processed):
        x = image.img_to_array(img2be_processed)
        x = np.expand_dims(x, axis=0)
        x = VGG16.prepare_input(x)
        prediction = self.extract_feature_model.predict(x)
        prediction_vector = []
        # make vectore of prediction
        for pred in prediction[0]:
            prediction_vector.append(pred)
        l2_norm = norm(prediction_vector)
        prediction_normalized_vector = np.divide(prediction_vector, l2_norm)
        return prediction_normalized_vector
