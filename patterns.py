#!/usr/bin/python3

import os
import re
import numpy
import shutil
import pickle
import logging
import argparse
from imutils import paths

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection

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

    def generate(self, regenerate=False):
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

        logging.info('Classification training')
        self.train_classifer()

        logging.info('Patterns saving')
        data = {
            'names': self.__names,
            'encodings': self.__encodings,
            'files': self.__files,
            'classifer': self.__classifer,
            'classes': self.__classes}
        dump = pickle.dumps(data)

        with open(self.__pickle_file, 'wb') as f:
            f.write(dump)

        logging.info(
            f'Patterns done: {self.__pickle_file} ({len(dump)} bytes)')

    def add_files(self, name, filenames, new=False):
        out_folder = os.path.join(self.__folder, name)
        if new:
            os.makedirs(out_folder, exist_ok=True)
        else:
            if not os.path.exists(out_folder):
                raise Exception(f"Name {name} not exists")
        for filename in filenames:
            out_filename = os.path.split(filename)[1]
            for n in self.__names:
                out_filename = re.sub(n + '_\d+_', '', out_filename)
                out_filename = re.sub(n + '_weak_\d+_', '', out_filename)
                out_filename = out_filename.replace(n, '')
            out_filename = re.sub('unknown_\d+_\d+_', '', out_filename)
            out_filename = os.path.join(out_folder, out_filename)
            logging.info(f'adding {filename} to {out_filename}')
            shutil.copyfile(filename, out_filename)

    def load(self):
        data = pickle.loads(open(self.__pickle_file, 'rb').read())
        self.__encodings = data['encodings']
        self.__names = data['names']
        self.__files = data['files']
        self.__classifer = data['classifer']
        self.__classes = data['classes']

    def train_classifer(self):
        RANDOM_SEED = 42

        data = numpy.array(self.__encodings)
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(self.__names)

        (x_train, x_test, y_train, y_test) = model_selection.train_test_split(
            data,
            labels,
            test_size=0.20,
            random_state=RANDOM_SEED)

        skf = model_selection.StratifiedKFold(n_splits=5)
        cv = skf.split(x_train, y_train)

        Cs = [0.001, 0.01, 0.1, 1, 10, 100]
        gammas = [0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = [
            {'C': Cs, 'kernel': ['linear']},
            {'C': Cs, 'gamma': gammas, 'kernel': ['rbf']}]

        init_est = svm.SVC(
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_SEED)

        grid_search = model_selection.GridSearchCV(
            estimator=init_est,
            param_grid=param_grid,
            n_jobs=4,
            cv=cv)

        grid_search.fit(x_train, y_train)

        self.__classifer = grid_search.best_estimator_
        self.__classes = list(le.classes_)

        y_pred = self.__classifer.predict(x_test)

        logging.info('Confusion matrix \n' +
                     str(metrics.confusion_matrix(y_test, y_pred)))

        logging.info('Classification report \n' +
                     str(metrics.classification_report(
                         y_test, y_pred, target_names=self.__classes)))

    def classifer(self):
        return self.__classifer

    def classes(self):
        return self.__classes

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
    parser.add_argument('files', nargs='*', help='Files with one face')
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
        patt.add_files(args.name, args.files)
    elif args.action == 'add_new':
        patt.load()
        patt.add_files(args.name, args.files, True)
    elif args.action == 'add_gen':
        patt.load()
        patt.add_files(args.name, args.files)
        patt.generate(args.regenerate)
    elif args.action == 'list':
        patt.load()
        for name, filename in zip(patt.names(), patt.files()):
            print(name, filename)


if __name__ == '__main__':
    main()
