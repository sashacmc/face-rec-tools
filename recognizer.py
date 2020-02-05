#!/usr/bin/python3

import io
import os
import cv2
import sys
import dlib
import numpy
import random
import shutil
import logging
import argparse
import threading
import itertools
import collections
import face_recognition
import concurrent.futures

from imutils import paths

import log
import recdb
import tools
import config
import cachedb
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
                 encoding_model='large',
                 max_workers=1,
                 cdb=None,
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
        self.__encoding_model = encoding_model
        self.__cdb = cdb
        self.__nearest_match = nearest_match

        self.__status = {'state': '', 'count': 0, 'current': 0}
        self.__status_lock = threading.Lock()

        self.__max_workers = int(max_workers)
        self.__executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.__max_workers)
        self.__pattern_encodings = numpy.array_split(
            numpy.array(self.__patterns.encodings()),
            self.__max_workers)

        self.__video_batch_size = 8 

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
            elif self.__method == 'match':
                self.match(*self.__args)
            elif self.__method == 'clusterize':
                self.clusterize(*self.__args)
            elif self.__method == 'save_faces':
                self.save_faces(*self.__args)
            logging.info(f'Thread done: {self.__method}')
            self.__status_state('done')
        except Exception as ex:
            logging.exception(ex)
            self.__status_state('error')

    def recognize_image(self, filename,
                        draw_landmarks=False):
        logging.info(f'recognize image: {filename}')

        image = tools.LazyImage(filename, self.__max_size)

        encoded_faces = self.encode_faces(image.get())

        self.__match_faces(encoded_faces)

        if draw_landmarks:
            self.__draw_landmarks(encoded_faces, image.get())

        return encoded_faces, image

    def recognize_video(self, filename):
        logging.info(f'recognize video: {filename}')
        video = cv2.VideoCapture(filename)
        ret = True
        count = 0
        batched_encoded_faces = []
        while ret:
            frames = []
            while len(frames) < self.__video_batch_size:
                ret, frame = video.read()
                if ret:
                    frame = tools.prepare_image(frame, self.__max_size)
                    frames.append(frame)
                else:
                    break

            batched_boxes = face_recognition.batch_face_locations(
                frames, batch_size=len(frames))
            count += len(frames)

            for image, boxes in zip(frames, batched_boxes):
                encodings = face_recognition.face_encodings(
                    image, boxes, self.__num_jitters, self.__encoding_model)

                encoded_faces = [{'encoding': e, 'box': b}
                                 for e, b in zip(encodings, boxes)]

                self.__match_faces(encoded_faces)
                for face in encoded_faces:
                    if face['dist'] < self.__threshold:
                        batched_encoded_faces.append(face)

        logging.info(f'done {count} frames: {filename}')
        return batched_encoded_faces

    def calc_names_in_video(self, encoded_faces):
        dct = collections.defaultdict(int)
        for face in encoded_faces:
            dct[face['name']] += 1
        return dct

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
            image, filtered_boxes, self.__num_jitters, self.__encoding_model)

        res = [{'encoding': e, 'box': b}
               for e, b in zip(encodings, filtered_boxes)]

        return res

    def __match_face_by_nearest(self, encoding):
        res = [r for r in self.__executor.map(face_recognition.face_distance,
                                              self.__pattern_encodings,
                                              itertools.repeat(encoding))]
        distances = numpy.concatenate(res)
        i = numpy.argmin(distances)
        return (distances[i], self.__patterns.names()[i])

    def __match_face_by_class(self, encoding):
        proba = self.__patterns.classifer().predict_proba(
            encoding.reshape(1, -1))[0]

        j = numpy.argmax(proba)
        dist = 1 - proba[j]
        return dist, self.__patterns.classes()[j]

    def __match_faces(self, encoded_faces):
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

    def __clusterize(self, files_faces, debug_out_folder=None):
        encs = []
        indexes = list(range(len(files_faces)))
        random.shuffle(indexes)
        for i in indexes:
            for j in range(len(files_faces[i]['faces'])):
                encs.append(dlib.vector(
                    files_faces[i]['faces'][j]['encoding']))

        labels = dlib.chinese_whispers_clustering(
            encs, self.__threshold_clusterize)

        labels = self.__reassign_by_count(labels)
        lnum = 0
        self.__status_count(len(files_faces))
        for i in indexes:
            self.__status_step()
            for j in range(len(files_faces[i]['faces'])):
                files_faces[i]['faces'][j]['name'] = \
                    'unknown_{:04d}'.format(labels[lnum])
                lnum += 1

            if debug_out_folder:
                filename = files_faces[i]['filename']
                image = tools.LazyImage(filename, self.__max_size)
                debug_out_file_name = self.__extract_filename(filename)
                self.__save_debug_images(
                    files_faces[i]['faces'], image,
                    debug_out_folder, debug_out_file_name)

    def recognize_files(self, filenames, db, debug_out_folder):
        self.__make_debug_out_folder(debug_out_folder)

        self.__status_count(len(filenames))
        for f in filenames:
            self.__status_step()
            try:
                encoded_faces, image = self.recognize_image(f)
                db.insert(f, encoded_faces, commit=False)
                if debug_out_folder:
                    debug_out_file_name = self.__extract_filename(f)
                    self.__save_debug_images(
                        encoded_faces, image,
                        debug_out_folder, debug_out_file_name)
            except Exception as ex:
                logging.exception(f'Image {f} recognition failed')
        db.commit()
        if self.__cdb is not None:
            self.__cdb.commit()

    def __get_files_faces_by_filter(self, db, fltr):
        tp = fltr['type']
        if tp == 'unmatched':
            return db.get_unmatched()
        elif tp == 'all':
            return db.get_all()
        elif tp == 'weak':
            return db.get_weak(fltr['path'])
        elif tp == 'weak_unmatched':
            return db.get_weak_unmatched(fltr['path'])
        elif tp == 'folder':
            return db.get_folder(fltr['path'])
        elif tp == 'name':
            return db.get_by_name(fltr['path'], fltr['name'])
        else:
            raise Exception(f'Unknown filter type: {tp}')

    def clusterize(self, db, fltr, debug_out_folder):
        self.__status_state('clusterize')
        files_faces = self.__get_files_faces_by_filter(db, fltr)
        self.__clusterize(files_faces, debug_out_folder)

    def match(self, db, fltr, debug_out_folder, save_all_faces):
        self.__status_state('match')
        files_faces = self.__get_files_faces_by_filter(db, fltr)
        self.__match_files_faces(files_faces, db,
                                 debug_out_folder, save_all_faces)

    def save_faces(self, db, fltr, debug_out_folder):
        self.__status_state('save_faces')
        files_faces = self.__get_files_faces_by_filter(db, fltr)
        self.__save_faces(files_faces, debug_out_folder)

    def __match_files_faces(
            self, files_faces, db, debug_out_folder, save_all_faces=False):
        cnt_all = 0
        cnt_changed = 0
        self.__status_count(len(files_faces))
        for ff in files_faces:
            self.__status_step()
            logging.info(f"match image: {ff['filename']}")
            self.__match_faces(ff['faces'])
            for face in ff['faces']:
                cnt_all += 1
                changed = False
                if 'oldname' in face and face['oldname'] != face['name']:
                    db.set_name(face['face_id'], face['name'], face['dist'],
                                commit=False)
                    cnt_changed += 1
                    changed = True
                    logging.info(
                        f"face {face['face_id']} in file '{ff['filename']}' " +
                        f"changed '{face['oldname']}' -> '{face['name']}'")
                if debug_out_folder and (changed or save_all_faces):
                    filename = ff['filename']
                    image = tools.LazyImage(filename, self.__max_size)
                    debug_out_file_name = self.__extract_filename(filename)
                    self.__save_debug_images(
                        (face,), image,
                        debug_out_folder, debug_out_file_name)
        db.commit()
        if self.__cdb is not None:
            self.__cdb.commit()
        logging.info(f'match done: count: {cnt_all}, changed: {cnt_changed}')

    def __save_faces(self, files_faces, debug_out_folder):
        self.__status_state('save_faces')
        self.__status_count(len(files_faces))
        for ff in files_faces:
            self.__status_step()
            filename = ff['filename']
            logging.info(f"save faces from image: {filename}")
            image = tools.LazyImage(filename, self.__max_size)
            debug_out_file_name = self.__extract_filename(filename)
            self.__save_debug_images(
                ff['faces'], image,
                debug_out_folder, debug_out_file_name)
        if self.__cdb is not None:
            self.__cdb.commit()

    def recognize_folder(self, folder, db, debug_out_folder, reencode=False):
        self.__status_state('recognize_folder')
        filenames = self.__get_images_from_folders(folder)

        if not reencode:
            filenames = set(filenames) - set(db.get_files(folder))

        if debug_out_folder is None:
            debug_out_folder = os.path.join(folder, 'tags')

        self.recognize_files(filenames, db, debug_out_folder)

    def remove_folder(self, folder, db):
        self.__status_state('remove_folder')
        files_faces = db.get_folder(folder)
        for ff in files_faces:
            logging.info(f"remove image: {ff['filename']}")
            db.remove(ff['filename'], False)
            if self.__cdb is not None:
                for face in ff['faces']:
                    self.__cdb.remove_face(face['face_id'])
        # delete files without faces
        files = db.get_files(folder)
        for f in files:
            logging.info(f"remove image: {f}")
            db.remove(f, False)
        db.commit()
        if self.__cdb is not None:
            self.__cdb.commit()

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
            out_folder = os.path.join(debug_out_folder, name)

            top, right, bottom, left = enc['box']

            prefix = '{}_{:03d}'.format(name, int(enc['dist'] * 100))
            out_filename = os.path.join(
                out_folder,
                f'{prefix}_{debug_out_file_name}_{left}x{top}.jpg')

            if self.__cdb is not None:
                if not self.__cdb.check_face(enc['face_id']):
                    out_stream = io.BytesIO()
                    tools.save_face(out_stream, image.get(),
                                    enc['box'], enc['encoding'],
                                    self.__debug_out_image_size)
                    self.__cdb.save_face(enc['face_id'],
                                         out_stream.getvalue())
                    logging.debug(f"face {enc['face_id']} cached")

                self.__cdb.add_to_cache(enc['face_id'], out_filename)
            else:
                self.__make_debug_out_folder(out_folder)
                tools.save_face(out_filename, image.get(),
                                enc['box'], enc['encoding'],
                                self.__debug_out_image_size)
                logging.debug(f'face saved to: {out_filename}')

    def __draw_landmarks(self, encoded_faces, image):
        boxes = [enc['box'] for enc in encoded_faces]
        landmarks = face_recognition.face_landmarks(
            image,
            face_locations=boxes,
            model=self.__encoding_model)

        for landmark in landmarks:
            for pts in landmark.values():
                for pt in pts:
                    cv2.circle(image, pt, 5, (0, 0, 255), -1)

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


def createRecognizer(patt, cfg, cdb):
    return Recognizer(patt,
                      model=cfg['main']['model'],
                      num_jitters=cfg['main']['num_jitters'],
                      threshold=cfg['main']['threshold'],
                      threshold_weak=cfg['main']['threshold_weak'],
                      threshold_clusterize=cfg['main']['threshold_clusterize'],
                      max_image_size=cfg['main']['max_image_size'],
                      min_face_size=cfg['main']['min_face_size'],
                      debug_out_image_size=cfg['main']['debug_out_image_size'],
                      encoding_model=cfg['main']['encoding_model'],
                      max_workers=cfg['main']['max_workers'],
                      cdb=cdb)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['recognize_image',
                 'recognize_video',
                 'recognize_folder',
                 'remove_folder',
                 'match_unmatched',
                 'match_all',
                 'match_folder',
                 'clusterize_unmatched',
                 'save_faces'])
    parser.add_argument('-p', '--patterns', help='Patterns file')
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-i', '--input', help='Input file or folder')
    parser.add_argument('-o', '--output', help='Output folder for faces')
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('-d', '--dry-run', help='Do''t modify DB',
                        action='store_true')
    parser.add_argument('-r', '--reencode', help='Reencode existing files',
                        action='store_true')
    return parser.parse_args()


def main():
    args = args_parse()

    cfg = config.Config(args.config)

    log.initLogger(args.logfile)

    if args.output and os.path.exists(args.output):
        shutil.rmtree(args.output)

    patt = patterns.Patterns(cfg.get_def('main', 'patterns', args.patterns),
                             model=cfg['main']['model'],
                             num_jitters=cfg['main']['num_jitters'],
                             encoding_model=cfg['main']['encoding_model'])
    patt.load()

    cachedb_file = cfg['main']['cachedb']
    if cachedb_file:
        cdb = cachedb.CacheDB(cachedb_file)
    else:
        cdb = None

    rec = createRecognizer(patt, cfg, cdb)

    db = recdb.RecDB(cfg['main']['db'], args.dry_run)

    if args.action == 'recognize_image':
        print(rec.recognize_image(args.input)[0])
    elif args.action == 'recognize_video':
        print(rec.calc_names_in_video(rec.recognize_video(args.input)))
    elif args.action == 'recognize_folder':
        rec.recognize_folder(args.input, db, args.output, args.reencode)
    elif args.action == 'remove_folder':
        rec.remove_folder(args.input, db)
    elif args.action == 'match_unmatched':
        rec.match(db, {'type': 'unmatched'}, args.output, False)
    elif args.action == 'match_all':
        rec.match(db, {'type': 'all'}, args.output, False)
    elif args.action == 'match_folder':
        rec.match(
            db, {'type': 'folder', 'path': args.input}, args.output, True)
    elif args.action == 'clusterize_unmatched':
        rec.clusterize(db, {'type': 'unmatched'}, args.output)
    elif args.action == 'save_faces':
        rec.save_faces(db, {'type': 'folder', 'path': args.input}, args.output)


if __name__ == '__main__':
    main()
