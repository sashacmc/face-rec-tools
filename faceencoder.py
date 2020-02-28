#!/usr/bin/python3

import logging

import cv2
import numpy as np
import tensorflow as tf
import face_recognition
from tensorflow.keras.preprocessing import image as keras_image
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from deepface.commons import distance as dst


class FaceEncoder(object):
    def __init__(self,
                 encoding_model='large',
                 distance_metric='default',
                 num_jitters=1):
        logging.info(
            f"Using {encoding_model} model and {distance_metric} metric")

        self.__tensorflow_init()

        self.__encoding_model = encoding_model
        self.__num_jitters = num_jitters
        if encoding_model == 'small':
            self.__encode = self.__encode_face_recognition
        elif encoding_model == 'large':
            self.__encode = self.__encode_face_recognition
        elif encoding_model == 'VGG-Face':
            self.__model = VGGFace.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (224, 224)
        elif encoding_model == 'OpenFace':
            self.__model = OpenFace.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (96, 96)
        elif encoding_model == 'Facenet':
            self.__model = Facenet.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (160, 160)
        elif encoding_model == 'DeepFace':
            self.__model = FbDeepFace.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (152, 152)
        else:
            raise ValueError("Invalid model_name: ", encoding_model)

        if distance_metric == 'default':
            self.__distance = face_recognition.face_distance
        elif distance_metric == 'cosine':
            self.__distance = lambda encodings, encoding: \
                [dst.findCosineDistance(e, encoding) for e in encodings]
        elif distance_metric == 'euclidean':
            self.__distance = lambda encodings, encoding: \
                [dst.findEuclideanDistance(e, encoding) for e in encodings]
        elif distance_metric == 'euclidean_l2':
            self.__distance = lambda encodings, encoding: \
                [dst.findEuclideanDistance(dst.l2_normalize(e),
                                           dst.l2_normalize(encoding))
                    for e in encodings]
        else:
            raise ValueError("Invalid distance_metric: ", distance_metric)

    def __tensorflow_init(self):
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        if tf.config.experimental.get_memory_growth(gpu):
            return
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=1536)])

    def __encode_deepface(self, image, boxes):
        res = []
        for box in boxes:
            top, right, bottom, left = box
            sub_image = image[top:bottom, left:right]
            sub_image = cv2.resize(sub_image, self.__input_shape)
            img_pixels = keras_image.img_to_array(sub_image)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            res.append(self.__model.predict(img_pixels)[0, :])
        return res

    def __encode_face_recognition(self, image, boxes):
        return face_recognition.face_encodings(
            image, boxes, self.__num_jitters,
            model=self.__encoding_model)

    def encode(self, image, boxes):
        return self.__encode(image, boxes)

    def distance(self, encoding1, encoding2):
        return self.__distance(encoding1, encoding2)
