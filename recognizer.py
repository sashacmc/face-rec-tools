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
import collections
import face_recognition

from PIL import Image

import log
import recdb
import tools
import config
import patterns


class Recognizer(object):
    def __init__(self, patterns, model='hog', num_jitters=1, threshold=0.5,
                 nearest_match=False):

        self.__patterns = patterns
        self.__model = model
        self.__num_jitters = int(num_jitters)
        self.__threshold = float(threshold)
        self.__threshold_weak = 0.5
        self.__threshold_clusterize = 0.5
        self.__max_size = 1000
        self.__min_size = 20
        self.__nearest_match = nearest_match

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

    def clusterize(self, files_faces, debug_out_folder=None):
        encs = []
        for i in range(len(files_faces)):
            for j in range(len(files_faces[i]['faces'])):
                encs.append(dlib.vector(
                    files_faces[i]['faces'][j]['encoding']))

        labels = dlib.chinese_whispers_clustering(
            encs, self.__threshold_clusterize)

        lnum = 0
        for i in range(len(files_faces)):
            for j in range(len(files_faces[i]['faces'])):
                if files_faces[i]['faces'][j]['name'] == '':
                    files_faces[i]['faces'][j]['name'] = \
                        'unknown_{:03d}'.format(labels[lnum])
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

        for f in filenames:
            res = self.recognize_image(f, debug_out_folder)
            db.insert(f, res)
            db.print_details(f)

    def clusterize_unmatched(self, db, debug_out_folder):
        files_faces = db.get_unmatched()
        self.clusterize(files_faces, debug_out_folder)

    def match_unmatched(self, db, debug_out_folder):
        files_faces = db.get_unmatched()
        cnt_all = 0
        cnt_changed = 0
        for ff in files_faces:
            logging.info(f"match image: {ff['filename']}")
            self.match(ff['faces'])
            for face in ff['faces']:
                cnt_all += 1
                if face['name']:
                    db.set_name(face['face_id'], face['name'], face['dist'])
                    cnt_changed += 1
                    logging.info(
                        f"face {face['face_id']} in file '{ff['filename']}' " +
                        f"matched to '{face['name']}'")
                    if debug_out_folder:
                        filename = ff['filename']
                        image = tools.read_image(filename, self.__max_size)
                        debug_out_file_name = self.__extract_filename(filename)
                        self.__save_debug_images(
                            (face,), image,
                            debug_out_folder, debug_out_file_name)
        logging.info(f'match_unmatched: {cnt_all}, changed: {cnt_changed}')

    def match_all(self, db, debug_out_folder):
        files_faces = db.get_all()
        cnt_all = 0
        cnt_changed = 0
        for ff in files_faces:
            logging.info(f"match image: {ff['filename']}")
            self.match(ff['faces'])
            for face in ff['faces']:
                cnt_all += 1
                if 'oldname' in face and face['oldname'] != face['name']:
                    db.set_name(face['face_id'], face['name'], face['dist'])
                    cnt_changed += 1
                    logging.info(
                        f"face {face['face_id']} in file '{ff['filename']}' " +
                        f"changed '{face['oldname']}' -> '{face['name']}'")
                    if debug_out_folder:
                        filename = ff['filename']
                        image = tools.read_image(filename, self.__max_size)
                        debug_out_file_name = self.__extract_filename(filename)
                        self.__save_debug_images(
                            (face,), image,
                            debug_out_folder, debug_out_file_name)
        logging.info(f'match_all: {cnt_all}, changed: {cnt_changed}')

    def save_faces(self, folder, db, debug_out_folder):
        filenames = self.__get_images_from_folders(folder)

        for filename in filenames:
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
        filenames = self.__get_images_from_folders(folder)

        if debug_out_folder is None:
            debug_out_folder = os.path.join(folder, 'tags')

        self.recognize_files(filenames, db, debug_out_folder)

    def __get_images_from_folders(self, folder):
        filenames = []
        for filename in os.listdir(folder):
            if os.path.splitext(filename)[1].lower() == '.jpg':
                filenames.append(os.path.join(folder, filename))
        return filenames

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
            if name == '':
                name = 'unknown_000'
            out_folder = os.path.join(debug_out_folder, name)
            self.__make_debug_out_folder(out_folder)

            top, right, bottom, left = enc['box']
            d = (bottom - top) // 2
            out_image = image[
                max(0, top - d):bottom + d,
                max(0, left - d):right + d]

            prefix = '{}_{:03d}'.format(name, int(enc['dist'] * 100))
            out_filename = os.path.join(
                out_folder, f'{prefix}_{debug_out_file_name}_{i}.jpg')

            encd = pickle.dumps(enc['encoding'])
            exif = piexif.dump(
                {"0th": {piexif.ImageIFD.ImageDescription: encd}})
            im = Image.fromarray(out_image)
            im.save(out_filename, exif=exif)

            logging.debug(f'face saved to: {out_filename}')


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
    parser.add_argument('-n', '--nearest-match',
                        help='Use nearest match (otherwise by classifier)',
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
                     cfg['main']['model'],
                     cfg['main']['num_jitters'],
                     cfg.get_def('main', 'threshold', args.threshold),
                     args.nearest_match)

    db = recdb.RecDB(cfg['main']['db'], args.dry_run)

    if args.action == 'recognize_image':
        print(rec.recognize_image(args.input, args.output))
    elif args.action == 'recognize_folder':
        rec.recognize_folder(args.input, db, args.output)
    elif args.action == 'match_unmatched':
        rec.match_unmatched(db, args.output)
    elif args.action == 'match_all':
        rec.match_all(db, args.output)
    elif args.action == 'clusterize_unmatched':
        rec.clusterize_unmatched(db, args.output)
    elif args.action == 'save_faces':
        rec.save_faces(args.input, db, args.output)


if __name__ == '__main__':
    main()
