#!/usr/bin/python3

import io
import os
import cv2
import dlib
import numpy
import random
import shutil
import logging
import argparse
import threading
import itertools
import collections
import concurrent.futures

try:
    import faceencoder
    import face_recognition
except Exception as ex:
    print('face_recognition not loaded, readonly mode: ' + str(ex))

from imutils import paths

import log
import recdb
import tools
import config
import cachedb
import patterns


SKIP_FACE = 'skip_face'


class Recognizer(threading.Thread):
    def __init__(self,
                 patts,
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
        self.__patterns = patts
        self.__model = model
        self.__encoder = faceencoder.FaceEncoder(
            encoding_model=encoding_model,
            distance_metric='cosine',
            num_jitters=num_jitters,
            align=True)
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

        self.__pattern_encodings = []
        self.__pattern_names = []
        for tp in (patterns.PATTERN_TYPE_BAD,
                   patterns.PATTERN_TYPE_GOOD,
                   patterns.PATTERN_TYPE_OTHER):
            encodings, names, files = self.__patterns.encodings(tp)
            self.__pattern_encodings.append(numpy.array_split(
                numpy.array(encodings),
                self.__max_workers))
            self.__pattern_names.append(names)

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
            elif self.__method == 'get_faces_by_face':
                self.get_faces_by_face(*self.__args)
            logging.info(f'Thread done: {self.__method}')
            self.__status_state('done')
        except Exception as ex:
            logging.exception(ex)
            self.__status_state('error')

    def recognize_image(self, filename):
        logging.info(f'recognize image: {filename}')

        image = tools.LazyImage(filename, self.__max_size)

        encoded_faces = self.encode_faces(image.get())

        self.__save_landmarks(encoded_faces, image.get())

        self.__match_faces(encoded_faces, False)

        return encoded_faces, image

    def reencode_image(self, filename, encoded_faces):
        logging.info(f'reencode image: {filename}')

        image = tools.LazyImage(filename, self.__max_size)

        boxes = [f['box'] for f in encoded_faces]
        encodings = self.__encoder.encode(image.get(), boxes)

        for i in range(len(encoded_faces)):
            encoded_faces[i] = encodings[i]

        self.__save_landmarks(encoded_faces, image.get())

    def recognize_video(self, filename):
        logging.info(f'recognize video: {filename}')
        video = tools.LazyVideo(filename, self.__max_size)
        frame_num = 0
        batched_encoded_faces = []
        while frame_num < len(video.frames()):
            frames = video.frames()[frame_num:
                                    frame_num + self.__video_batch_size]

            batched_boxes = face_recognition.batch_face_locations(
                frames, batch_size=len(frames))

            for image, boxes in zip(frames, batched_boxes):
                encodings = self.__encoder.encode(image, boxes)
                encoded_faces = [{'encoding': e, 'box': b, 'frame': frame_num}
                                 for e, b in zip(encodings, boxes)]

                self.__match_faces(encoded_faces, True)
                self.__save_landmarks(encoded_faces, image)
                for face in encoded_faces:
                    if face['dist'] >= self.__threshold:
                        face['name'] = SKIP_FACE
                batched_encoded_faces += encoded_faces
                frame_num += 1

        logging.info(f'done {frame_num} frames: {filename}')
        return batched_encoded_faces, video

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

        encodings = self.__encoder.encode(image, filtered_boxes)
        res = [{'encoding': e, 'box': b, 'frame': 0}
               for e, b in zip(encodings, filtered_boxes)]

        return res

    def __match_face_by_nearest(self, encoding, tp):
        res = [r for r in self.__executor.map(self.__encoder.distance,
                                              self.__pattern_encodings[tp],
                                              itertools.repeat(encoding))]
        distances = numpy.concatenate(res)
        if len(distances) == 0:
            return 1, ''
        i = numpy.argmin(distances)
        return (distances[i], self.__pattern_names[tp][i])

    def __match_faces(self, encoded_faces, good_only):
        if len(self.__pattern_encodings) == 0:
            logging.warning('Empty patterns')

        for i in range(len(encoded_faces)):
            encoding = encoded_faces[i]['encoding']
            dist, name = self.__match_face_by_nearest(
                encoding, patterns.PATTERN_TYPE_GOOD)
            if not good_only:
                dist_bad, name_bad = self.__match_face_by_nearest(
                    encoding, patterns.PATTERN_TYPE_BAD)
                # dist_other, name_other = self.__match_face_by_nearest(
                #    encoding, patterns.PATTERN_TYPE_OTHER)
                dist_other, name_other = 1, ''
                if dist_other < dist_bad:
                    dist_bad = dist_other
                    name_bad = name_other
                if dist_bad < dist:
                    name = name_bad + '_bad'
                    dist = dist_bad

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
                media = tools.load_media(filename, self.__max_size)
                debug_out_file_name = self.__extract_filename(filename)
                self.__save_debug_images(
                    files_faces[i]['faces'], media,
                    debug_out_folder, debug_out_file_name)

    def recognize_files(self, filenames, db, debug_out_folder):
        self.__make_debug_out_folder(debug_out_folder)

        self.__status_count(len(filenames))
        for f in filenames:
            self.__status_step()
            try:
                ext = os.path.splitext(f)[1].lower()
                if ext in tools.IMAGE_EXTS:
                    encoded_faces, media = self.recognize_image(f)
                elif ext in tools.VIDEO_EXTS:
                    encoded_faces, media = self.recognize_video(f)
                else:
                    logging.warning(f'Unknown ext: {ext}')
                    continue
                db.insert(f, encoded_faces, commit=False)
                if debug_out_folder:
                    debug_out_file_name = self.__extract_filename(f)
                    self.__save_debug_images(
                        encoded_faces, media,
                        debug_out_folder, debug_out_file_name)
            except Exception as ex:
                logging.exception(f'Image {f} recognition failed')
        db.commit()
        if self.__cdb is not None:
            self.__cdb.commit()

    def reencode_files(self, files_faces, db):
        for ff in files_faces:
            try:
                encoded_faces = ff['faces']
                filename = ff['filename']
                ext = os.path.splitext(filename)[1].lower()
                if ext in tools.IMAGE_EXTS:
                    self.reencode_image(filename, encoded_faces)
                elif ext in tools.VIDEO_EXTS:
                    encoded_faces = self.recognize_video(filename)
                else:
                    logging.warning(f'Unknown ext: {ext}')
                    continue
                db.insert(filename, encoded_faces, commit=False)
            except Exception as ex:
                logging.exception(f'{filename} reencoding failed')
        db.commit()

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
            self.__match_faces(ff['faces'], False)
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
                    media = tools.load_media(filename, self.__max_size)
                    debug_out_file_name = self.__extract_filename(filename)
                    self.__save_debug_images(
                        (face,), media,
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
            media = tools.load_media(filename, self.__max_size)
            debug_out_file_name = self.__extract_filename(filename)
            self.__save_debug_images(
                ff['faces'], media,
                debug_out_folder, debug_out_file_name)
        if self.__cdb is not None:
            self.__cdb.commit()

    def recognize_folder(self, folder, db, debug_out_folder, reencode=False):
        self.__status_state('recognize_folder')
        filenames = self.__get_media_from_folder(folder)

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

    def __get_media_from_folder(self, folder):
        return list(paths.list_files(
            folder,
            validExts=tools.IMAGE_EXTS + tools.VIDEO_EXTS))

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
            self, encoded_faces, media, debug_out_folder, debug_out_file_name):

        for enc in encoded_faces:
            name = enc['name']
            if name == SKIP_FACE:
                continue
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
                    if not enc['landmarks']:
                        self.__save_landmarks((enc,), media.get(enc['frame']))
                    tools.save_face(out_stream, media.get(enc['frame']), enc,
                                    self.__debug_out_image_size,
                                    media.filename())
                    self.__cdb.save_face(enc['face_id'],
                                         out_stream.getvalue())
                    logging.debug(f"face {enc['face_id']} cached")

                self.__cdb.add_to_cache(enc['face_id'], out_filename)
            else:
                self.__make_debug_out_folder(out_folder)
                if not enc['landmarks']:
                    self.__save_landmarks((enc,), media.get(enc['frame']))
                tools.save_face(out_filename, media.get(enc['frame']), enc,
                                self.__debug_out_image_size,
                                media.filename())
                logging.debug(f'face saved to: {out_filename}')

    def __save_landmarks(self, encoded_faces, image):
        if self.__encoding_model in ('small', 'large'):
            boxes = [enc['box'] for enc in encoded_faces]
            landmarks = face_recognition.face_landmarks(
                image,
                face_locations=boxes,
                model=self.__encoding_model)

            for i in range(len(encoded_faces)):
                encoded_faces[i]['landmarks'] = landmarks[i]
        else:
            for i in range(len(encoded_faces)):
                encoded_faces[i]['landmarks'] = {}

    def get_faces_by_face(self, db, filename, debug_out_folder,
                          remove_file=False):
        logging.info(f'get faces by face: {filename}')
        self.__status_state('get_faces_by_face')

        image = tools.LazyImage(filename, self.__max_size)

        encoded_faces = self.encode_faces(image.get())
        face = encoded_faces[0]
        logging.debug(f'found face: {face}')

        all_encodings = db.get_all_encodings(self.__max_workers)

        res = [r for r in self.__executor.map(
            self.__encoder.distance,
            all_encodings[0],
            itertools.repeat(face['encoding']))]
        distances = numpy.concatenate(res)

        filtered = []
        for dist, info in zip(distances, all_encodings[1]):
            if dist < 0.4:
                filtered.append((dist, info))
        filtered.sort()

        logging.debug(f'{len(filtered)} faces matched')

        self.__status_count(len(filtered))
        for dist, info in filtered:
            self.__status_step()
            fname, face = info
            face['dist'] = dist
            media = tools.load_media(fname, self.__max_size)
            debug_out_file_name = self.__extract_filename(fname)
            self.__save_debug_images(
                (face,), media,
                debug_out_folder, debug_out_file_name)
        if remove_file:
            logging.debug(f'removing temp file: {filename}')
            os.remove(filename)

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


def createRecognizer(patt, cfg, cdb=None):
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
                 'save_faces',
                 'get_faces_by_face'])
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
        print(rec.calc_names_in_video(rec.recognize_video(args.input)[0]))
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
    elif args.action == 'get_faces_by_face':
        rec.get_faces_by_face(db, args.input, args.output)


if __name__ == '__main__':
    main()
