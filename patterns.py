#!/usr/bin/python3

import os
import pickle
import logging
import face_recognition
from imutils import paths

import tools


class Patterns(object):
    def __init__(self, folder, model='hog', max_size=1000):
        self.__folder = folder
        self.__pickle_file = os.path.join(folder, 'patterns.pickle')
        self.__encodings = []
        self.__names = []
        self.__files = []
        self.__model = model
        self.__max_size = max_size

    def generate(self):
        logging.info(f'Patterns generation: {self.__folder}')
        image_files = list(paths.list_images(self.__folder))

        for (i, image_file) in enumerate(image_files):
            name = image_file.split(os.path.sep)[-2]
            logging.info(f'{i + 1}/{len(image_files)} file: {image_file}')

            image = tools.read_image(image_file, self.__max_size)

            boxes = face_recognition.face_locations(image, model=self.__model)
            if len(boxes) != 1:
                logging.warning(
                    f'Multiple or zero faces detected in {image_file}. Skip.')
                continue
            encodings = face_recognition.face_encodings(image, boxes)

            self.__encodings.append(encodings[0])
            self.__names.append(name)
            self.__files.append(os.path.split(image_file)[1])

        logging.info('Patterns saving')
        data = {
            'names': self.__names,
            'encodings': self.__encodings,
            'files': self.__files}
        dump = pickle.dumps(data)

        with open(self.__pickle_file, 'wb') as f:
            f.write(dump)

        logging.info(
            f'Patterns done: {self.__pickle_file} ({len(dump)} bytes)')

    def load(self):
        data = pickle.loads(open(self.__pickle_file, 'rb').read())
        self.__encodings = data['encodings']
        self.__names = data['names']
        self.__files = data['files']

    def encodings(self):
        return self.__encodings

    def names(self):
        return self.__names

    def files(self):
        return self.__files


if __name__ == '__main__':
    import sys
    import log

    log.initLogger()

    patt = Patterns(sys.argv[1], 'cnn')
    patt.generate()
