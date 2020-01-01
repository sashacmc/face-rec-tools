#!/usr/bin/python3

import os
import cv2
import sys
import dlib
import numpy
import shutil
import piexif
import pickle
import logging
import argparse
import threading
import collections
import face_recognition

from PIL import Image
from imutils import paths

import log
import recdb
import tools
import config
import patterns


class Recognizer(threading.Thread):
    def __init__(self,
                 patterns,
                 model='hog',
                 num_jitters=1,
                 threshold=0.3,
                 threshold_weak=0.5,
                 threshold_clusterize=0.5,
                 max_image_size=1000,
                 min_face_size=20,
                 debug_out_image_size=100,
                 nearest_match=True):

        threading.Thread.__init__(self)
        self.__patterns = patterns
        self.__model = model
        self.__num_jitters = int(num_jitters)
        self.__threshold = float(threshold)
        self.__threshold_weak = float(threshold_weak)
        self.__threshold_clusterize = float(threshold_clusterize)
        self.__max_size = int(max_image_size)
        self.__min_size = int(min_face_size)
        self.__debug_out_image_size = int(debug_out_image_size)
        self.__nearest_match = nearest_match
        self.__status = {'state': '', 'count': 0, 'current': 0}
        self.__status_lock = threading.Lock()

    def start_method(self, method, *args):
        self.__status_state(method)
        self.__method = method
        self.__args = args
        self.start()

    def run(self):
        try:
            logging.info(f'Run in thread: {self.__method}{self.__args}')
            if self.__method == 'recognize_folder':
                self.recognize_folder(*self.__args)
            elif self.__method == 'match_unmatched':
                self.match_unmatched(*self.__args)
            elif self.__method == 'match_all':
                self.match_all(*self.__args)
            elif self.__method == 'match_folder':
                self.match_folder(*self.__args)
            elif self.__method == 'clusterize_unmatched':
                self.clusterize_unmatched(*self.__args)
            elif self.__method == 'save_faces':
                self.save_faces(*self.__args)
            logging.info(f'Thread done: {self.__method}')
            self.__status_state('done')
        except Exception as ex:
            logging.exception(ex)
            self.__status_state('error')

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
        if not boxes:
            return []

        filtered_boxes = []
        for box in boxes:
            (top, right, bottom, left) = box
            face_image = image[top:bottom, left:right]
            (height, width) = face_image.shape[:2]
            if height < self.__min_size or width < self.__min_size:
                logging.debug(f'Too small face: {height}x{width}')
                continue
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            if fm < 50:
                logging.debug(f'Too blurry face: {fm}')
                continue
            filtered_boxes.append(box)

        encodings = face_recognition.face_encodings(
            image, filtered_boxes, self.__num_jitters)

        res = [{'encoding': e, 'box': b}
               for e, b in zip(encodings, filtered_boxes)]

        return res

    def __match_face_by_nearest(self, encoding):
        distances = face_recognition.face_distance(
            self.__patterns.encodings(), encoding)

        names = [(dist, name)
                 for dist, name in zip(distances, self.__patterns.names())]
        names.sort()
        return names[0]

    def __match_face_by_class(self, encoding):
        proba = self.__patterns.classifer().predict_proba(
            encoding.reshape(1, -1))[0]

        j = numpy.argmax(proba)
        dist = 1 - proba[j]
        return dist, self.__patterns.classes()[j]

    def match(self, encoded_faces):
        if len(self.__patterns.encodings()) == 0:
            logging.warning('Empty patterns')

        for i in range(len(encoded_faces)):
            encoding = encoded_faces[i]['encoding']
            if self.__nearest_match:
                dist, name = self.__match_face_by_nearest(encoding)
            else:
                dist, name = self.__match_face_by_class(encoding)

            logging.debug(f'matched: {name}: {dist}')
            if 'name' in encoded_faces[i]:
                encoded_faces[i]['oldname'] = encoded_faces[i]['name']

            if dist < self.__threshold:
                pass
            elif dist < self.__threshold_weak:
                name += '_weak'
            else:
                name = ''
                dist = 1

            encoded_faces[i]['name'] = name
            encoded_faces[i]['dist'] = dist

    def __reassign_by_count(self, labels):
        dct = collections.defaultdict(int)
        for old_label in labels:
            dct[old_label] += 1
        lst = [(count, old_label) for old_label, count in dct.items()]
        lst.sort(reverse=True)
        trans = {count_old_label[1]: new_label for new_label,
                 count_old_label in enumerate(lst)}
        return [trans[old_label] for old_label in labels]

    def clusterize(self, files_faces, debug_out_folder=None):
        encs = []
        for i in range(len(files_faces)):
            for j in range(len(files_faces[i]['faces'])):
                encs.append(dlib.vector(
                    files_faces[i]['faces'][j]['encoding']))

        labels = dlib.chinese_whispers_clustering(
            encs, self.__threshold_clusterize)

        labels = self.__reassign_by_count(labels)
        lnum = 0
        self.__status_count(len(files_faces))
        for i in range(len(files_faces)):
            self.__status_step()
            for j in range(len(files_faces[i]['faces'])):
                if files_faces[i]['faces'][j]['name'] == '':
                    files_faces[i]['faces'][j]['name'] = \
                        'unknown_{:04d}'.format(labels[lnum])
                lnum += 1

            if debug_out_folder:
                filename = files_faces[i]['filename']
                image = tools.read_image(filename, self.__max_size)
                debug_out_file_name = self.__extract_filename(filename)
                self.__save_debug_images(
                    files_faces[i]['faces'], image,
                    debug_out_folder, debug_out_file_name)

    def recognize_files(self, filenames, db, debug_out_folder):
        self.__make_debug_out_folder(debug_out_folder)

        self.__status_count(len(filenames))
        for f in filenames:
            self.__status_step()
            res = self.recognize_image(f, debug_out_folder)
            db.insert(f, res)
            db.print_details(f)

    def clusterize_unmatched(self, db, debug_out_folder):
        self.__status_state('clusterize_unmatched')
        files_faces = db.get_unmatched()
        self.clusterize(files_faces, debug_out_folder)

    def match_unmatched(self, db, debug_out_folder):
        self.__status_state('match_unmatched')
        files_faces = db.get_unmatched()
        self.__match_files_faces(files_faces, db, debug_out_folder)

    def match_all(self, db, debug_out_folder):
        self.__status_state('match_all')
        files_faces = db.get_all()
        self.__match_files_faces(files_faces, db, debug_out_folder)

    def match_folder(self, folder, db, debug_out_folder, save_all_faces=True):
        self.__status_state('match_folder')
        files_faces = db.get_folder(folder)
        self.__match_files_faces(
            files_faces, db, debug_out_folder, save_all_faces)

    def __match_files_faces(
            self, files_faces, db, debug_out_folder, save_all_faces=False):
        cnt_all = 0
        cnt_changed = 0
        self.__status_count(len(files_faces))
        for ff in files_faces:
            self.__status_step()
            logging.info(f"match image: {ff['filename']}")
            self.match(ff['faces'])
            for face in ff['faces']:
                cnt_all += 1
                changed = False
                if 'oldname' in face and face['oldname'] != face['name']:
                    db.set_name(face['face_id'], face['name'], face['dist'])
                    cnt_changed += 1
                    changed = True
                    logging.info(
                        f"face {face['face_id']} in file '{ff['filename']}' " +
                        f"changed '{face['oldname']}' -> '{face['name']}'")
                if debug_out_folder and (changed or save_all_faces):
                    filename = ff['filename']
                    image = tools.read_image(filename, self.__max_size)
                    debug_out_file_name = self.__extract_filename(filename)
                    self.__save_debug_images(
                        (face,), image,
                        debug_out_folder, debug_out_file_name)
        logging.info(f'match done: count: {cnt_all}, changed: {cnt_changed}')

    def save_faces(self, folder, db, debug_out_folder):
        self.__status_state('save_faces')
        filenames = self.__get_images_from_folders(folder)

        self.__status_count(len(filenames))
        for filename in filenames:
            self.__status_step()
            logging.info(f"save faces from image: {filename}")
            files_faces = db.get_faces(filename)
            if len(files_faces) == 0:
                continue
            filename = files_faces[0]['filename']
            image = tools.read_image(filename, self.__max_size)
            debug_out_file_name = self.__extract_filename(filename)
            self.__save_debug_images(
                files_faces[0]['faces'], image,
                debug_out_folder, debug_out_file_name)
            logging.info(f"face: {debug_out_file_name}")

    def recognize_folder(self, folder, db, debug_out_folder):
        self.__status_state('recognize_folder')
        filenames = self.__get_images_from_folders(folder)

        if debug_out_folder is None:
            debug_out_folder = os.path.join(folder, 'tags')

        self.recognize_files(filenames, db, debug_out_folder)

    def __get_images_from_folders(self, folder):
        return list(paths.list_images(folder))

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

        for enc in encoded_faces:
            name = enc['name']
            if name == '':
                name = 'unknown_000'
            alg = '_' + enc['alg'] if 'alg' in enc else ''
            out_folder = os.path.join(debug_out_folder, name)
            self.__make_debug_out_folder(out_folder)

            top, right, bottom, left = enc['box']
            d = (bottom - top) // 2
            out_image = image[
                max(0, top - d):bottom + d,
                max(0, left - d):right + d]

            prefix = '{}_{:03d}'.format(name, int(enc['dist'] * 100))
            out_filename = os.path.join(
                out_folder,
                f'{prefix}_{debug_out_file_name}_{left}x{top}{alg}.jpg')

            encd = pickle.dumps(enc['encoding'], protocol=0)
            exif = piexif.dump(
                {"0th": {piexif.ImageIFD.ImageDescription: encd}})
            im = Image.fromarray(out_image)
            im.thumbnail((self.__debug_out_image_size,
                          self.__debug_out_image_size))
            im.save(out_filename, exif=exif)

            logging.debug(f'face saved to: {out_filename}')

    def status(self):
        with self.__status_lock:
            return self.__status

    def __status_state(self, cmd):
        with self.__status_lock:
            self.__status['state'] = cmd

    def __status_count(self, count):
        with self.__status_lock:
            self.__status['count'] = count
            self.__status['current'] = 0

    def __status_step(self):
        with self.__status_lock:
            self.__status['current'] += 1


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['recognize_image',
                 'recognize_folder',
                 'match_unmatched',
                 'match_all',
                 'clusterize_unmatched',
                 'save_faces'])
    parser.add_argument('-p', '--patterns', help='Patterns file')
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-i', '--input', help='Input file or folder')
    parser.add_argument('-o', '--output', help='Output folder for faces')
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('--threshold', help='Match threshold')
    parser.add_argument('-d', '--dry-run', help='Do''t modify DB',
                        action='store_true')
    return parser.parse_args()


def main():
    args = args_parse()

    cfg = config.Config(args.config)

    log.initLogger(args.logfile)

    if args.output and os.path.exists(args.output):
        shutil.rmtree(args.output)

    patt = patterns.Patterns(cfg.get_def('main', 'patterns', args.patterns),
                             cfg['main']['model'])
    patt.load()

    rec = Recognizer(patt,
                     model=cfg['main']['model'],
                     num_jitters=cfg['main']['num_jitters'],
                     threshold=cfg.get_def(
                         'main', 'threshold', args.threshold),
                     threshold_weak=cfg['main']['threshold_weak'],
                     threshold_clusterize=cfg['main']['threshold_clusterize'],
                     max_image_size=cfg['main']['max_image_size'],
                     min_face_size=cfg['main']['min_face_size'],
                     debug_out_image_size=cfg['main']['debug_out_image_size'])

    db = recdb.RecDB(cfg['main']['db'], args.dry_run)

    if args.action == 'recognize_image':
        print(rec.recognize_image(args.input, args.output))
    elif args.action == 'recognize_folder':
        rec.recognize_folder(args.input, db, args.output)
    elif args.action == 'match_unmatched':
        rec.match_unmatched(db, args.output)
    elif args.action == 'match_all':
        rec.match_all(db, args.output)
    elif args.action == 'match_folder':
        rec.match_folder(args.input, db, args.output)
    elif args.action == 'clusterize_unmatched':
        rec.clusterize_unmatched(db, args.output)
    elif args.action == 'save_faces':
        rec.save_faces(args.input, db, args.output)


if __name__ == '__main__':
    main()
