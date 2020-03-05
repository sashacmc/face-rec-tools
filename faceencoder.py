#!/usr/bin/python3

import logging

import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
import face_alignment
import face_recognition
from tensorflow.keras.preprocessing import image as keras_image
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from deepface.commons import distance as dst

PRED_TYPES = {'face': slice(0, 17),
              'eyebrow1': slice(17, 22),
              'eyebrow2': slice(22, 27),
              'nose': slice(27, 31),
              'nostril': slice(31, 36),
              'eye1': slice(36, 42),
              'eye2': slice(42, 48),
              'lips': slice(48, 60),
              'teeth': slice(60, 68)
              }


class FaceEncoder(object):
    def __init__(self,
                 encoding_model='large',
                 distance_metric='default',
                 num_jitters=1,
                 align=False,
                 debug_out_folder='/mnt/multimedia/tmp/test/'):

        logging.info(
            f"Using {encoding_model} model and {distance_metric} metric")

        self.__tensorflow_init()

        self.__encoding_model = encoding_model
        self.__num_jitters = int(num_jitters)
        self.__debug_out_folder = debug_out_folder
        self.__debug_out_counter = 0

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

        if align:
            self.__aligner = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D,
                device='cuda',
                flip_input=True)
        else:
            self.__aligner = None

    def __tensorflow_init(self):
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        if tf.config.experimental.get_memory_growth(gpu):
            return
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=1536)])

    def __save_debug(self, image):
        if self.__debug_out_folder is None:
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.__debug_out_folder,
                                 str(self.__debug_out_counter) + '.jpg'),
                    image)
        self.__debug_out_counter += 1

    def __align(self, image, box, pred):
        desiredLeftEye = (0.35, 0.35)
        desiredFaceWidth, desiredFaceHeight = self.__input_shape

        leftEyePts = pred[PRED_TYPES['eye2']]
        rightEyePts = pred[PRED_TYPES['eye1']]

        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        output = cv2.warpAffine(image, M, self.__input_shape,
                                flags=cv2.INTER_CUBIC)
        return output

    def __aligner_boxes(self, boxes):
        return [(left, top, right, bottom)
                for top, right, bottom, left in boxes]

    def __convert_to_landmarks(self, preds):
        if preds is None:
            return None
        return [{
            "chin": pred[PRED_TYPES['face']].tolist(),
            "left_eyebrow": pred[PRED_TYPES['eyebrow2']].tolist(),
            "right_eyebrow": pred[PRED_TYPES['eyebrow1']].tolist(),
            "nose_bridge": pred[PRED_TYPES['nose']].tolist(),
            "nose_tip": pred[PRED_TYPES['nostril']].tolist(),
            "left_eye": pred[PRED_TYPES['eye2']].tolist(),
            "right_eye": pred[PRED_TYPES['eye1']].tolist(),
            "top_lip": pred[PRED_TYPES['lips']].tolist(),
            "bottom_lip": pred[PRED_TYPES['teeth']].tolist()}
            for pred in preds]

    def __encode_deepface(self, image, boxes):
        res = []
        if self.__aligner is not None:
            preds = self.__aligner.get_landmarks_from_image(
                image, self.__aligner_boxes(boxes))
            landmarks = self.__convert_to_landmarks(preds)
        else:
            preds = [None] * len(boxes)
            landmarks = [{}] * len(boxes)

        for box, pred in zip(boxes, preds):
            if self.__aligner is not None:
                sub_image = self.__align(
                    image, self.__aligner_boxes((box,))[0], pred)
                self.__save_debug(sub_image)
            else:
                top, right, bottom, left = box
                sub_image = image[top:bottom, left:right]
                sub_image = cv2.resize(sub_image, self.__input_shape)
                self.__save_debug(sub_image)
            img_pixels = keras_image.img_to_array(sub_image)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            res.append(self.__model.predict(img_pixels)[0, :])
        return res, landmarks

    def __create_full_object_detection(self, box, fa_landmarks):
        rect = dlib.rectangle(*box)
        points = [dlib.point(*fa_point) for fa_point in fa_landmarks]
        return dlib.full_object_detection(rect, points)

    def __encode_face_recognition(self, image, boxes):
        if self.__aligner is not None:
            aboxes = self.__aligner_boxes(boxes)
            preds = self.__aligner.get_landmarks_from_image(image, aboxes)
            raw_landmarks = [self.__create_full_object_detection(box, pred)
                             for box, pred in zip(aboxes, preds)]
        else:
            raw_landmarks = None

        encodings = face_recognition.face_encodings(
            image, boxes, self.__num_jitters,
            model=self.__encoding_model,
            raw_landmarks=raw_landmarks)

        landmarks = face_recognition.face_landmarks(
            image,
            face_locations=boxes,
            model=self.__encoding_model,
            raw_landmarks=raw_landmarks)
        return encodings, landmarks

    def encode(self, image, boxes):
        return self.__encode(image, boxes)

    def distance(self, encodings, encoding):
        return self.__distance(encodings, encoding)
