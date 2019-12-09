#!/usr/bin/python3

import os
import re
import glob
import shutil
import pickle
import logging
import argparse
from imutils import paths

import log
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

    def generate(self, regenerate):
        import face_recognition

        logging.info(f'Patterns generation: {self.__folder} ({regenerate})')
        image_files = list(paths.list_images(self.__folder))

        if not regenerate:
            self.load()
            filtered = []
            for image_file in image_files:
                if os.path.split(image_file)[1] not in self.__files:
                    filtered.append(image_file)
            image_files = filtered

        for (i, image_file) in enumerate(image_files):
            name = image_file.split(os.path.sep)[-2]
            logging.info(f'{i + 1}/{len(image_files)} file: {image_file}')

            image = tools.read_image(image_file, self.__max_size)

            boxes = face_recognition.face_locations(image, model=self.__model)
            if len(boxes) != 1:
                logging.warning(
                    f'{len(boxes)} faces detected in {image_file}. Skip.')
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

    def add_files(self, name, filepatt, new=False):
        out_folder = os.path.join(self.__folder, name)
        if new:
            os.makedirs(out_folder, exist_ok=True)
        else:
            if not os.path.exists(out_folder):
                raise Exception(f"Name {name} not exists")
        for filename in glob.glob(filepatt):
            out_filename = os.path.split(filename)[1]
            for n in self.__names:
                out_filename = out_filename.replace(n, '')
            out_filename = re.sub('_unknown_\d+', '', out_filename)
            out_filename = os.path.join(out_folder, out_filename)
            logging.info(f'adding {filename} to {out_filename}')
            shutil.copyfile(filename, out_filename)

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


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['gen',
                 'add',
                 'add_new',
                 'add_gen',
                 'list'])
    parser.add_argument('-p', '--patterns', help='Patterns file')
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-n', '--name', help='Person name')
    parser.add_argument('-f', '--file', help='Files with one face')
    parser.add_argument('-r', '--regenerate', help='Regenerate all',
                        action='store_true')
    return parser.parse_args()


def main():
    args = args_parse()
    log.initLogger(args.logfile)

    patt = Patterns(args.patterns, 'cnn')

    if args.action == 'gen':
        patt.generate(args.regenerate)
    elif args.action == 'add':
        patt.load()
        patt.add_files(args.name, args.file)
    elif args.action == 'add_new':
        patt.load()
        patt.add_files(args.name, args.file, True)
    elif args.action == 'add_gen':
        patt.load()
        patt.add_files(args.name, args.file)
        patt.generate(args.regenerate)
    elif args.action == 'list':
        patt.load()
        for name, filename in zip(patt.names(), patt.files()):
            print(name, filename)


if __name__ == '__main__':
    main()
