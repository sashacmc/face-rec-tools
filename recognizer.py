#!/usr/bin/python3

import os
import cv2
import logging
import collections
import face_recognition

import recdb
import tools
import patterns


class Recognizer(object):
    def __init__(self, patterns, model='hog', num_jitters=1, threshold=0.4):
        self.__patterns = patterns
        self.__model = model
        self.__num_jitters = num_jitters
        self.__threshold = threshold
        self.__max_size = 1000

    def recognize_image(self, filename, debug_out_folder=None):
        logging.info(f'recognize image: {filename}')

        image = tools.read_image(filename, self.__max_size)

        encoded_faces = self.encode_faces(image)

        self.match(encoded_faces)

        if debug_out_folder:
            debug_out_file_name = self.__extract_filename(filename)
            self.__save_debug_images(
                encoded_faces, image, debug_out_folder, debug_out_file_name)

        return encoded_faces

    def encode_faces(self, image):

        boxes = face_recognition.face_locations(image, model=self.__model)

        encodings = face_recognition.face_encodings(
            image, boxes, self.__num_jitters)

        res = [{'encoding': e, 'box': b}
               for e, b in zip(encodings, boxes)]

        return res

    def match(self, encoded_faces):
        for i in range(len(encoded_faces)):
            distances = face_recognition.face_distance(
                self.__patterns.encodings(), encoded_faces[i]['encoding'])

            names = collections.defaultdict(lambda: [0, 0.])
            for j, name in enumerate(self.__patterns.names()):
                if distances[j] <= self.__threshold:
                    names[name][0] += 1
                    names[name][1] += distances[j]

            names_mid = []
            for name in names:
                names_mid.append((names[name][1] / names[name][0], name))

            if len(names_mid) != 0:
                names_mid.sort()
                dist, name = names_mid[0]
                logging.info(f'found: {name}: dist: {dist}')
            else:
                name = ''

            encoded_faces[i]['name'] = name

    def recognize_files(self, filenames, db, debug_out_folder):
        self.__make_debug_out_folder(debug_out_folder)

        for f in filenames:
            res = self.recognize_image(f, debug_out_folder)
            db.insert(f, res)
            db.print_details(f)

    def recognize_folder(self, folder, db):
        filenames = []
        for filename in os.listdir(folder):
            if os.path.splitext(filename)[1].lower() == '.jpg':
                filenames.append(os.path.join(folder, filename))

        self.recognize_files(
            filenames, db, os.path.join(folder, 'tags'))

    def __make_debug_out_folder(self, debug_out_folder):
        if debug_out_folder:
            try:
                os.makedirs(debug_out_folder, exist_ok=False)

                with open(
                    os.path.join(
                        debug_out_folder, '.plexignore'), 'w') as f:

                    f.write('*\n')

            except FileExistsError:
                pass

    def __extract_filename(self, filename):
        return os.path.splitext(os.path.split(filename)[1])[0]

    def __save_debug_images(
            self, encoded_faces, image, debug_out_folder, debug_out_file_name):

        for i, enc in enumerate(encoded_faces):
            name = enc['name']
            out_folder = os.path.join(debug_out_folder, name)
            self.__make_debug_out_folder(out_folder)

            top, right, bottom, left = enc['box']
            d = (bottom - top) // 4
            out_image = image[top - d:bottom + d, left - d:right + d]
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

            out_filename = os.path.join(
                out_folder, f'{debug_out_file_name}_{i}_{name}.jpg')

            cv2.imwrite(out_filename, out_image)
            logging.debug(f'face saved to: {out_filename}')


if __name__ == '__main__':
    import sys
    import log

    log.initLogger()

    patt = patterns.Patterns(sys.argv[1])
    patt.load()

    rec = Recognizer(patt, 'cnn')
    db = recdb.RecDB('rec.db')
    # print(rec.recognize_image(sys.argv[2], 'test_out'))
    rec.recognize_folder(sys.argv[2], db)
