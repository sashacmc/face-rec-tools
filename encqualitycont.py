#!/usr/bin/python3

import os
import shutil
import logging
import argparse

import cv2
import numpy as np
from imutils import paths

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

import log
import config
import patterns


class EncodingQualityCont(object):
    def __init__(self, patterns, modelfile):
        self.__patterns = patterns
        self.__modelfile = modelfile
        self.__epoch_size = 40
        self.__batch_size = 1000
        self.__test_size = 0
        self.__image_size = 50
        self.__cut_width = 25

    def __load_image(self, path):
        try:
            image = cv2.imread(path)
            image = image[self.__cut_width: -self.__cut_width,
                          self.__cut_width: -self.__cut_width]
            image = cv2.resize(image, (self.__image_size, self.__image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = img_to_array(image) / 255.
        except Exception:
            logging.exception(f'Problem with image {path}')
            raise
        return image

    def train(self):
        self.__patterns.load()
        logging.info('train started')

        files_bad = self.__patterns.encodings(patterns.PATTERN_TYPE_BAD)[2]
        logging.info(f'found {len(files_bad)} bad files')

        files_good = self.__patterns.encodings(patterns.PATTERN_TYPE_GOOD)[2]
        logging.info(f'found {len(files_good)} good files')

        limit = min(len(files_bad), len(files_good))
        files_bad = files_bad[:limit]
        files_good = files_good[:limit]

        labels_bad = [0, ] * len(files_bad)
        labels_good = [1, ] * len(files_good)

        files = files_bad + files_good
        labels = labels_bad + labels_good

        images = np.array([self.__load_image(f) for f in files])
        labels = np.asarray(labels).astype('float32').reshape((-1, 1))
        # labels = to_categorical(labels, 2)

        logging.info(f'loaded {len(images)} images')
        if self.__test_size != 0:
            images, images_test, labels, labels_test = train_test_split(
                images, labels, test_size=self.__test_size,
                shuffle=True, random_state=42)
            dataset_test = tf.data.Dataset.from_tensor_slices(
                (images_test, labels_test))
            dataset_test = dataset_test.batch(1)
        else:
            dataset_test = None
            images_test = []

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            horizontal_flip=True)

        datagen.fit(images)

        logging.info(f'ready for train {len(images)} images')
        logging.info(f'ready for test {len(images_test)} images')

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(
                self.__image_size, self.__image_size, 1)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        """
        model = keras.Sequential([
            keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                                input_shape=(self.__image_size,
                                             self.__image_size, 3)),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1)
        ])
        model = keras.Sequential([
            keras.layers.Conv2D(16, 1, padding='same', activation='relu',
                                input_shape=(self.__image_size,
                                             self.__image_size, 1)),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(32, 1, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 1, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2)
        ])
        """
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        model.summary()
        logging.info('fit started')

        model.fit(datagen.flow(images, labels, batch_size=self.__batch_size),
                  validation_data=dataset_test,
                  validation_steps=len(images_test),
                  steps_per_epoch=len(images) / self.__batch_size,
                  epochs=self.__epoch_size)

        if dataset_test is not None:
            results = model.evaluate(images_test, labels_test,
                                     batch_size=self.__batch_size)
            logging.info(f'test loss, test acc: {results}')

        logging.info(f'model saving: {self.__modelfile}')
        model.save(self.__modelfile)

    def test(self, files):
        logging.info(f'model loading: {self.__modelfile}')
        model = keras.models.load_model(self.__modelfile)
        images = np.array([self.__load_image(f) for f in files])
        return model.predict(images, batch_size=self.__batch_size)

    def sort_patterns(self, files, out_folder):
        images = []
        for f in files:
            if os.path.isdir(f):
                images += list(paths.list_images(f))
            else:
                images.append(f)

        for path, pred in zip(images, self.test(images)):
            splitted = path.split(os.path.sep)
            name = splitted[-2]
            if name == 'bad':
                name = splitted[-3]

            res = pred[0]
            if res < 0.5:
                name += '_bad'

            outdir = os.path.join(out_folder, name)
            prefix = '{}_{:03d}_'.format(name, abs(int(res * 100)))
            outname = os.path.join(outdir, prefix + os.path.split(path)[1])
            print(path + ': ' + str(pred))
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            shutil.copyfile(path, outname)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['check',
                 'train',
                 'test',
                 'sort_patterns'])
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('files', nargs='*', help='Files with one face')
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('-o', '--output', help='Output folder for faces')
    parser.add_argument('-m', '--model', help='Model file',
                        default='EncodingQualityCont.h5')
    return parser.parse_args()


def main():
    args = args_parse()
    cfg = config.Config(args.config)
    log.initLogger(args.logfile)

    patt = patterns.Patterns(cfg['main']['patterns'],
                             model=cfg['main']['model'],
                             max_size=cfg['main']['max_image_size'],
                             num_jitters=cfg['main']['num_jitters'],
                             encoding_model=cfg['main']['encoding_model'])

    cont = EncodingQualityCont(patt, args.model)

    if args.action == 'train':
        cont.train()
    elif args.action == 'test':
        print(cont.test(args.files))
    elif args.action == 'sort_patterns':
        cont.sort_patterns(args.files, args.output)


if __name__ == '__main__':
    main()
