from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.utils import get_file
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as k, Input, Model
import numpy as np


VGG16_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGGFACE_DIR = 'model_scheme/vggface'
V1_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'
V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'


class VGG16:
    def __init__(self, include_top=True, input_shape=None, classes=2622):
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=48,
                                          data_format=k.image_data_format(),
                                          require_flatten=include_top)
        img_input = Input(input_shape)
        self.VGG16Model = self.cnn(img_input=img_input, classes=classes)

    @staticmethod
    def cnn(img_input, classes):
        # BLOCK 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        # BLOCK 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        # BLOCK 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

        # BLOCK 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

        # BLOCK 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

        # TOP LAYER
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(4096, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dense(classes, name='fc8')(x)
        x = Activation('softmax', name="fc8/softmax")(x)

        model = Model(img_input, x, name="vgg16")

        weights_path = get_file('rcmalli_vggface_tf_vgg16.h5', VGG16_WEIGHTS_PATH, cache_subdir=VGGFACE_DIR)
        model.load_weights(weights_path, by_name=True)
        return model

    def fc_7(self):
        fc7_out = self.VGG16Model.get_layer('fc7')
        return fc7_out

    @staticmethod
    def prepare_input(x, data_format=None):
        x_temp = np.copy(x)
        if data_format is None:
            data_format = k.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] -= 93.5940
        x_temp[..., 1] -= 104.7624
        x_temp[..., 2] -= 129.1863
        return x_temp
