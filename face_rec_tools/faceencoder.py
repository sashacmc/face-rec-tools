#!/usr/bin/python3

import os
import cv2
import sys
import math
import numpy as np
import face_recognition

sys.path.insert(0, os.path.abspath('..'))

from face_rec_tools import log  # noqa
from face_rec_tools import tools  # noqa


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
                 align=True,
                 debug_out_folder=None):

        log.info(f"Using {encoding_model} model and {distance_metric} metric")

        self.__encoding_model = encoding_model
        self.__num_jitters = int(num_jitters)
        self.__debug_out_folder = debug_out_folder
        self.__debug_out_counter = 0

        if encoding_model == 'small':
            self.__encode = self.__encode_face_recognition
            align = False
        elif encoding_model == 'large':
            self.__encode = self.__encode_face_recognition
        elif encoding_model == 'VGG-Face':
            from deepface.basemodels import VGGFace
            self.__model = VGGFace.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (224, 224)
        elif encoding_model == 'OpenFace':
            from deepface.basemodels import OpenFace
            self.__model = OpenFace.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (96, 96)
        elif encoding_model == 'Facenet':
            from deepface.basemodels import Facenet
            self.__model = Facenet.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (160, 160)
        elif encoding_model == 'DeepFace':
            from deepface.basemodels import FbDeepFace
            self.__model = FbDeepFace.loadModel()
            self.__encode = self.__encode_deepface
            self.__input_shape = (152, 152)
        else:
            raise ValueError("Invalid model_name: ", encoding_model)

        if distance_metric == 'default':
            self.__distance = face_recognition.face_distance
        elif distance_metric == 'cosine':
            from deepface.commons import distance
            self.__distance = lambda encodings, encoding: \
                [distance.findCosineDistance(e, encoding) for e in encodings]
        elif distance_metric == 'euclidean':
            from deepface.commons import distance
            self.__distance = lambda encodings, encoding: \
                [distance.findEuclideanDistance(
                    e, encoding) for e in encodings]
        elif distance_metric == 'euclidean_l2':
            from deepface.commons import distance as dst
            self.__distance = lambda encodings, encoding: \
                [dst.findEuclideanDistance(dst.l2_normalize(e),
                                           dst.l2_normalize(encoding))
                    for e in encodings]
        else:
            raise ValueError("Invalid distance_metric: ", distance_metric)

        if align:
            import face_alignment
            self.__aligner = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._3D,
                device='cuda' if tools.has_cuda() else 'cpu',
                flip_input=True)
        else:
            self.__aligner = None

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
            "chin": pred[PRED_TYPES['face']],
            "left_eyebrow": pred[PRED_TYPES['eyebrow2']],
            "right_eyebrow": pred[PRED_TYPES['eyebrow1']],
            "nose_bridge": pred[PRED_TYPES['nose']],
            "nose_tip": pred[PRED_TYPES['nostril']],
            "left_eye": pred[PRED_TYPES['eye2']],
            "right_eye": pred[PRED_TYPES['eye1']],
            "top_lip": pred[PRED_TYPES['lips']],
            "bottom_lip": pred[PRED_TYPES['teeth']]}
            for pred in preds]

    def __convert_to_2D(self, preds):
        if preds is None or len(preds) == 0:
            return None
        return [pred.astype(int).tolist()
                for pred in np.delete(preds, np.s_[2:], 2)]

    def __get_eyes_angle(self, pred):
        if pred is None:
            return None
        leftEyePts = pred[PRED_TYPES['eye2']]
        rightEyePts = pred[PRED_TYPES['eye1']]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        dist_2d = math.hypot(leftEyeCenter[0] - rightEyeCenter[0],
                             leftEyeCenter[1] - rightEyeCenter[1])
        dist_z = abs(leftEyeCenter[2] - rightEyeCenter[2])
        return float(np.degrees(np.arctan2(dist_z, dist_2d)))

    def __profile_angles(self, preds):
        return [self.__get_eyes_angle(pred) for pred in preds]

    def __encode_deepface(self, image, boxes):
        from tensorflow.keras.preprocessing import image as keras_image

        res = []
        if self.__aligner is not None:
            preds = self.__aligner.get_landmarks_from_image(
                image, self.__aligner_boxes(boxes))
            landmarks = self.__convert_to_landmarks(
                self.__convert_to_2D(preds))
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
        return res, landmarks, self.__profile_angles(preds)

    def __encode_face_recognition(self, image, boxes):
        if len(boxes) == 0:
            return [], [], []
        if self.__aligner is not None:
            preds = self.__aligner.get_landmarks_from_image(
                image, self.__aligner_boxes(boxes))
            preds2d = self.__convert_to_2D(preds)
        else:
            preds = [None] * len(boxes)
            preds2d = None

        encodings = face_recognition.face_encodings(
            image, boxes, self.__num_jitters,
            model=self.__encoding_model,
            landmark_points=preds2d)

        landmarks = face_recognition.face_landmarks(
            image,
            face_locations=boxes,
            model=self.__encoding_model,
            landmark_points=preds2d)

        return encodings, landmarks, self.__profile_angles(preds)

    def encode(self, image, boxes):
        return self.__encode(image, boxes)

    def distance(self, encodings, encoding):
        return self.__distance(encodings, encoding)
