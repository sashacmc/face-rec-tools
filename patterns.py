#!/usr/bin/python3

import os
import re
import shutil
import pickle
import logging
import argparse
import collections
from imutils import paths

import log
import tools
import config

FACE_FILENAME = '0_face.jpg'
BAD_FOLDERNAME = 'bad'
OTHER_FOLDERNAME = 'other'

PATTERN_TYPE_BAD = 0
PATTERN_TYPE_GOOD = 1
PATTERN_TYPE_OTHER = 2


class Patterns(object):
    FILES_ENC = 0
    FILES_NAME = 1
    FILES_TIME = 2
    FILES_TYPE = 3

    def __init__(self, folder, model='hog', max_size=1000,
                 num_jitters=1, encoding_model='large'):
        self.__folder = folder
        self.__pickle_file = os.path.join(folder, 'patterns.pickle')
        self.__files = {}
        self.__persons = []
        self.__model = model
        self.__encoding_model = encoding_model
        self.__max_size = int(max_size)
        self.__num_jitters = int(num_jitters)

    def generate(self, regenerate=False):
        try:
            import faceencoder
            import face_recognition
        except Exception:
            logging.exception('faceencoder not loaded, readonly mode')

        logging.info(f'Patterns generation: {self.__folder} ({regenerate})')

        encoder = faceencoder.FaceEncoder(
            encoding_model=self.__encoding_model,
            num_jitters=self.__num_jitters)

        image_files = {}
        for image_file in list(paths.list_images(self.__folder)):
            if os.path.split(image_file)[1] == FACE_FILENAME:
                continue
            image_files[image_file] = os.stat(image_file).st_mtime

        if not regenerate:
            self.load()
            filtered = {}
            for image_file in image_files:
                filename_exsists = image_file in self.__files
                if not filename_exsists or \
                   self.__files[image_file][self.FILES_TIME] != \
                   image_files[image_file]:

                    filtered[image_file] = image_files[image_file]
                    if filename_exsists:
                        self.__remove_file(image_file)

            if len(filtered) == 0:
                logging.info('Nothing changed')
                return

            image_files = filtered

        for (i, image_file) in enumerate(image_files):
            splitted = image_file.split(os.path.sep)
            name = splitted[-2]
            if name == BAD_FOLDERNAME:
                name = splitted[-3]
                tp = PATTERN_TYPE_BAD
            elif name == OTHER_FOLDERNAME:
                name = splitted[-3]
                tp = PATTERN_TYPE_OTHER
            else:
                tp = PATTERN_TYPE_GOOD

            logging.info(f'{i + 1}/{len(image_files)} file: {image_file}')

            descr, thumbnail = tools.load_face_description(image_file)
            try:
                encoding = descr['encoding']
            except Exception:
                encoding = None

            if encoding is None:
                image = tools.read_image(image_file, self.__max_size)

                boxes = face_recognition.face_locations(
                    image, model=self.__model)
                if len(boxes) != 1:
                    logging.warning(
                        f'{len(boxes)} faces detected in {image_file}. Skip.')
                    continue
                encodings = encoder.encode(image, boxes)
                encoding = encodings[0]

            self.__files[image_file] = [encoding,
                                        name,
                                        image_files[image_file],
                                        tp]
        self.__save()

    def __save(self):
        self.__persons = self.__calcPersons()

        logging.info('Patterns saving')
        data = {
            'files': self.__files,
            'persons': self.__persons}
        dump = pickle.dumps(data)

        with open(self.__pickle_file, 'wb') as f:
            f.write(dump)

        logging.info(
            f'Patterns done: {self.__pickle_file} ({len(dump)} bytes)')

    def __calc_out_filename(self, filename):
        out_filename = os.path.split(filename)[1]
        for person in reversed(sorted(self.__persons, key=len)):
            n = person['name']
            out_filename = re.sub(n + '_bad_weak_\d+_', '', out_filename)
            out_filename = re.sub(n + '_weak_\d+_', '', out_filename)
            out_filename = re.sub(n + '_bad_\d+_', '', out_filename)
            out_filename = re.sub(n + '_\d+_', '', out_filename)
            out_filename = out_filename.replace(n, '')
        out_filename = re.sub('unknown_\d+_\d+_', '', out_filename)
        return out_filename

    def __calc_filename(self, filename):
        if filename.startswith('http://'):
            path, filename = os.path.split(filename)
            name = os.path.split(path)[1]
            return os.path.join(self.__folder, name,
                                self.__calc_out_filename(filename))
        return filename

    def add_files(self, name, filenames, new=False, move=False, bad=False):
        out_folder = os.path.join(self.__folder, name)
        if bad:
            out_folder = os.path.join(out_folder, BAD_FOLDERNAME)
        if new:
            os.makedirs(out_folder, exist_ok=True)
        else:
            if not os.path.exists(out_folder):
                raise Exception(f"Name {name} not exists")
        for filename in filenames:
            out_filename = os.path.join(out_folder,
                                        self.__calc_out_filename(filename))
            logging.info(f'adding {filename} to {out_filename}')
            shutil.copyfile(filename, out_filename)
            if move:
                os.remove(filename)

    def add_file_data(self, name, filename, data, bad=False):
        out_folder = os.path.join(self.__folder, name)
        if bad:
            out_folder = os.path.join(out_folder, BAD_FOLDERNAME)
        os.makedirs(out_folder, exist_ok=True)
        out_filename = os.path.join(out_folder,
                                    self.__calc_out_filename(filename))
        logging.info(f'adding data of {filename} to {out_filename}')
        with open(out_filename, 'wb') as f:
            f.write(data)

    def __remove_file(self, filename):
        del self.__files[filename]
        logging.debug(f'File removed: {filename}')

    def remove_files(self, filenames):
        self.load()
        for filename in filenames:
            filename = self.__calc_filename(filename)
            self.__remove_file(filename)
            try:
                os.remove(filename)
            except FileNotFoundError:
                logging.exception('Skip file removing')
        self.__save()

    def load(self):
        try:
            data = pickle.loads(open(self.__pickle_file, 'rb').read())
            self.__files = data['files']
            self.__persons = data['persons']
        except Exception:
            logging.exception(f'Can''t load patterns: {self.__pickle_file}')

    def optimize(self):
        import dlib
        encodings, names, files = self.encodings()
        encs = [dlib.vector(enc) for enc in encodings]
        labels = dlib.chinese_whispers_clustering(encs, 0.1)
        uniq = {}
        to_remove = []
        for fname, label in zip(files, labels):
            if label not in uniq:
                uniq[label] = fname
            else:
                name1 = fname.split(os.path.sep)[-2]
                name2 = uniq[label].split(os.path.sep)[-2]
                if name1 != name2:
                    logging.warning(
                        f'Different person {fname} {uniq[label]}')
                else:
                    to_remove.append(fname)

        self.remove_files(to_remove)
        logging.info(f'Optimized from {len(encs)} to {len(uniq)}.')

    def analyze(self):
        fset = {}
        for f in self.__files:
            filename = os.path.split(f)[1]
            if filename in fset:
                logging.warning(
                    f'Duplicate pattern file: {f} ({fset[filename]})')
            else:
                fset[filename] = f

        dct = collections.defaultdict(list)
        for f, (enc, name, time, tp) in self.__files.items():
            dct[len(enc)].append(f)
        if len(dct) != 1:
            logging.warning('Inconsistent encoding: ' + str(dct.keys()))
            max_key = list(dct.keys())[0]
            for key in dct:
                if len(dct[max_key]) < len(dct[key]):
                    max_key = key
            del dct[max_key]
            for lst in dct.values():
                for f in lst:
                    logging.warning(f'wrong encoding: {f}')

    def __calcPersons(self):
        dct = collections.defaultdict(lambda: {'count': 0, 'image': 'Z'})
        for f in self.__files:
            name = self.__files[f][self.FILES_NAME]
            dct[name]['count'] += 1
            dct[name]['image'] = min(dct[name]['image'], f)

        for name in dct:
            face_filename = os.path.join(self.__folder, name, FACE_FILENAME)
            if os.path.exists(face_filename):
                dct[name]['image'] = face_filename

        res = [{'name': k, 'count': v['count'], 'image': v['image']}
               for k, v in dct.items()]
        res.sort(key=lambda el: el['count'], reverse=True)
        return res

    def encodings(self, ftp=None):
        encodings = []
        names = []
        files = []
        for f, (enc, name, time, tp) in self.__files.items():
            if ftp is None or ftp == tp:
                encodings.append(enc)
                names.append(name)
                files.append(f)

        return encodings, names, files

    def persons(self):
        return self.__persons


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['gen',
                 'add',
                 'add_new',
                 'add_gen',
                 'remove',
                 'list',
                 'persons',
                 'optimize',
                 'analyze',
                 'set_landmarks',
                 'clear_landmarks'])
    parser.add_argument('-p', '--patterns', help='Patterns file')
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-n', '--name', help='Person name')
    parser.add_argument('files', nargs='*', help='Files with one face')
    parser.add_argument('-r', '--regenerate', help='Regenerate all',
                        action='store_true')
    parser.add_argument('-c', '--config', help='Config file')
    return parser.parse_args()


def main():
    args = args_parse()
    cfg = config.Config(args.config)
    log.initLogger(args.logfile)

    patt = Patterns(cfg.get_def('main', 'patterns', args.patterns),
                    model=cfg['main']['model'],
                    max_size=cfg['main']['max_image_size'],
                    num_jitters=cfg['main']['num_jitters'],
                    encoding_model=cfg['main']['encoding_model'])

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
    elif args.action == 'remove':
        patt.remove_files(args.files)
    elif args.action == 'list':
        patt.load()
        for name, filename in zip(patt.names(), patt.files()):
            print(name, filename)
    elif args.action == 'persons':
        patt.load()
        for p in patt.persons():
            print(p)
    elif args.action == 'optimize':
        patt.load()
        patt.optimize()
    elif args.action == 'analyze':
        patt.load()
        patt.analyze()
    elif args.action == 'set_landmarks':
        for f in args.files:
            tools.enable_landmarks(f, True)
    elif args.action == 'clear_landmarks':
        for f in args.files:
            tools.enable_landmarks(f, False)


if __name__ == '__main__':
    main()
