import os

# from tensorflow_core.python.keras.utils.data_utils import Sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image as process_image
from keras.utils import Sequence
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.keras import Model
import tensorflow as tf

from openvino.runtime import Core

class DeepModel():
    '''MobileNet deep model.'''
    def __init__(self):
        self._model = self._define_model()

        print('Loading image-retrieval-0001.')
        print()

    @staticmethod
    def _define_model(output_layer=-1):
        '''Define a pre-trained MobileNet model.

        Args:
            output_layer: the number of layer that output.

        Returns:
            Class of keras model with weights.
        '''
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.40  # dynamically grow the memory used on the GPU
        # config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        # base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        core = Core()
        classification_model_xml = 'image-retrieval-0001/FP32/image-retrieval-0001.xml'
        model = core.read_model(model=classification_model_xml)
        compiled_model = core.compile_model(model=model, device_name="CPU")

        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        # output = base_model.layers[output_layer].output
        # output = GlobalAveragePooling2D()(output)
        model_result = Model(inputs=input_layer, outputs=output_layer)
        return model_result

    @staticmethod
    def preprocess_image(path):
        '''Process an image to numpy array.

        Args:
            path: the path of the image.

        Returns:
            Numpy array of the image.
        '''
        img = process_image.load_img(path, target_size=(224, 224))
        x = process_image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    @staticmethod
    def cosine_distance(input1, input2):
        '''Calculating the distance of two inputs.

        The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.

        Args:
            input1, input2: two input numpy arrays.

        Returns:
            Element-wise cosine distances of two inputs.
        '''
        # return np.dot(input1, input2) / (np.linalg.norm(input1) * np.linalg.norm(input2))
        return np.dot(input1, input2.T) / \
                np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
                        np.linalg.norm(input2.T, axis=0, keepdims=True))

    def extract_feature(self, img):
        '''Extract deep feature using MobileNet model.

        Args:
            generator: a predict generator inherit from `keras.utils.Sequence`.

        Returns:
            The output features of all inputs.
        '''
        features = self._model.predict(img, batch_size=1)
        return features


class DataSequence(Sequence):
    '''Predict generator inherit from `keras.utils.Sequence`.'''
    def __init__(self, paras, generation, batch_size=32):
        self.list_of_label_fields = []
        self.list_of_paras = paras
        self.data_generation = generation
        self.batch_size = batch_size
        self.__idx = 0

    def __len__(self):
        '''The number of batches per epoch.'''
        return int(np.ceil(len(self.list_of_paras) / self.batch_size))

    def __getitem__(self, idx):
        '''Generate one batch of data.'''
        paras = self.list_of_paras[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x, batch_fields = self.data_generation(paras)

        if idx == self.__idx:
            self.list_of_label_fields += batch_fields
            self.__idx += 1

        return np.array(batch_x)