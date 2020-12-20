#!/usr/bin/python3

import io
import os
import cv2
import sys
import dlib
import time
import numpy
import random
import shutil
import signal
import logging
import argparse
import itertools
import collections
import concurrent.futures

sys.path.insert(0, os.path.abspath('..'))

try:
    import face_recognition
    from face_rec_tools import faceencoder
except Exception as ex:
    print('face_recognition not loaded, readonly mode: ' + str(ex))

from face_rec_tools import log  # noqa
from face_rec_tools import recdb  # noqa
from face_rec_tools import tools  # noqa
from face_rec_tools import config  # noqa
from face_rec_tools import cachedb  # noqa
from face_rec_tools import patterns  # noqa


class Recognizer(object):
    def __init__(self,
                 patts,
                 model='hog',
                 num_jitters=1,
                 threshold=0.3,
                 threshold_weak=0.5,
                 threshold_clusterize=0.5,
                 threshold_equal=0.1,
                 max_image_size=1000,
                 max_video_frames=180,
                 video_frames_step=1,
                 min_face_size=20,
                 max_face_profile_angle=90,
                 min_video_face_count=10,
                 debug_out_image_size=100,
                 encoding_model='large',
                 distance_metric='default',
                 max_workers=1,
                 video_batch_size=1,
                 nomedia_files=(),
                 cdb=None,
                 db=None,
                 status=None):

        self.__patterns = patts
        self.__model = model
        self.__encoder = faceencoder.FaceEncoder(
            encoding_model=encoding_model,
            distance_metric=distance_metric,
            num_jitters=num_jitters,
            align=True)
        self.__threshold = float(threshold)
        self.__threshold_weak = float(threshold_weak)
        self.__threshold_clusterize = float(threshold_clusterize)
        self.__threshold_equal = float(threshold_equal)
        self.__max_size = int(max_image_size)
        self.__max_video_frames = int(max_video_frames)
        self.__video_frames_step = int(video_frames_step)
        self.__min_size = int(min_face_size)
        self.__max_face_profile_angle = int(max_face_profile_angle)
        self.__min_video_face_count = int(min_video_face_count)
        self.__debug_out_image_size = int(debug_out_image_size)
        self.__encoding_model = encoding_model
        self.__nomedia_files = nomedia_files
        self.__cdb = cdb
        self.__db = db

        if status is None:
            self.__status = {'state': '', 'stop': False,
                             'count': 0, 'current': 0}
        else:
            self.__status = status

        self.__max_workers = int(max_workers)
        self.__executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.__max_workers)

        self.__pattern_encodings = []
        self.__pattern_names = []
        self.__pattern_files = []
        for tp in (patterns.PATTERN_TYPE_BAD,
                   patterns.PATTERN_TYPE_GOOD,
                   patterns.PATTERN_TYPE_OTHER):
            encodings, names, files = self.__patterns.encodings(tp)
            self.__pattern_encodings.append(numpy.array_split(
                numpy.array(encodings),
                self.__max_workers))
            self.__pattern_names.append(names)
            self.__pattern_files.append(files)

        self.__video_batch_size = int(video_batch_size)

    def recognize_image(self, filename):
        logging.info(f'recognize image: {filename}')

        image = tools.LazyImage(filename, self.__max_size)

        encoded_faces = self.encode_faces(image.get())

        if self.__match_faces(encoded_faces):
            return encoded_faces, image
        else:
            return [], None

    def reencode_image(self, filename, encoded_faces):
        logging.info(f'reencode image: {filename}')

        image = tools.LazyImage(filename, self.__max_size)

        boxes = [f['box'] for f in encoded_faces]
        encodings, landmarks, profile_angles = self.__encoder.encode(
            image.get(), boxes)

        for i in range(len(encoded_faces)):
            encoded_faces[i]['encoding'] = encodings[i]
            encoded_faces[i]['landmarks'] = landmarks[i]
            encoded_faces[i]['profile_angle'] = profile_angles[i]

    def recognize_video(self, filename):
        logging.info(f'recognize video: {filename}')
        video = tools.LazyVideo(filename,
                                self.__max_size,
                                self.__max_video_frames,
                                self.__video_frames_step)

        all_frames = list(video.frames().items())

        batched_encoded_faces = []
        cnt = 0
        while cnt < len(all_frames):
            if self.__step_stage(step=0):
                return [], None

            frame_numbers, frames = zip(
                *all_frames[cnt: cnt + self.__video_batch_size])

            batched_boxes = face_recognition.batch_face_locations(
                list(frames), batch_size=len(frames))

            for image, boxes, frame_num in zip(frames,
                                               batched_boxes,
                                               frame_numbers):
                if self.__step_stage(step=0):
                    return [], None
                encodings, landmarks, profile_angles = self.__encoder.encode(
                    image, boxes)
                encoded_faces = [
                    {'encoding': e,
                     'box': b,
                     'frame': frame_num,
                     'landmarks': l,
                     'profile_angle': pa}
                    for e, l, b, pa in zip(encodings, landmarks,
                                           boxes, profile_angles)]
                encoded_faces = self.__filter_encoded_faces(encoded_faces)

                self.__match_faces(encoded_faces)
                batched_encoded_faces += encoded_faces
                cnt += 1

        logging.info(f'done {cnt} frames: {filename}')
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
                logging.debug(f'Skip too small face: {height}x{width}')
                continue
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            if fm < 50:
                logging.debug(f'Skip too blurry face: {fm}')
                continue
            filtered_boxes.append(box)

        if len(filtered_boxes):
            encodings, landmarks, profile_angles = self.__encoder.encode(
                image, filtered_boxes)
            res = [{'encoding': e,
                    'box': b,
                    'frame': 0,
                    'landmarks': l,
                    'profile_angle': pa}
                   for e, l, b, pa in zip(encodings, landmarks,
                                          filtered_boxes, profile_angles)]
            res = self.__filter_encoded_faces(res)
        else:
            res = []

        return res

    def __match_face_by_nearest(self, encoding, tp):
        res = [r for r in self.__executor.map(self.__encoder.distance,
                                              self.__pattern_encodings[tp],
                                              itertools.repeat(encoding))]
        distances = numpy.concatenate(res)
        if len(distances) == 0:
            return 1, '', ''
        i = numpy.argmin(distances)
        return (distances[i],
                self.__pattern_names[tp][i],
                self.__pattern_files[tp][i])

    def __match_faces(self, encoded_faces):
        if len(self.__pattern_encodings) == 0:
            logging.warning('Empty patterns')

        for i in range(len(encoded_faces)):
            if self.__step_stage(step=0):
                return False
            encoding = encoded_faces[i]['encoding']
            dist, name, pattern = self.__match_face_by_nearest(
                encoding, patterns.PATTERN_TYPE_GOOD)
            if dist > 0.001:  # skip zero match
                dist_bad, name_bad, pattern_bad = self.__match_face_by_nearest(
                    encoding, patterns.PATTERN_TYPE_BAD)
                # match to bad only equal faces
                if dist_bad < self.__threshold_equal and dist_bad < dist:
                    name = name_bad + '_bad'
                    dist = dist_bad
                    pattern = pattern_bad

            logging.debug(f'matched: {name}: {dist}: {pattern}')
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
            encoded_faces[i]['pattern'] = pattern
        return True

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
        self.__start_stage(len(files_faces))
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
        for i in indexes:
            if self.__step_stage():
                break
            for j in range(len(files_faces[i]['faces'])):
                files_faces[i]['faces'][j]['name'] = \
                    'unknown_{:05d}'.format(labels[lnum])
                lnum += 1

            if debug_out_folder:
                filename = files_faces[i]['filename']
                media = tools.load_media(filename,
                                         self.__max_size,
                                         self.__max_video_frames,
                                         self.__video_frames_step)
                debug_out_file_name = self.__extract_filename(filename)
                self.__save_debug_images(
                    files_faces[i]['faces'], media,
                    debug_out_folder, debug_out_file_name)
        self.__end_stage()

    def recognize_files(self, filenames, debug_out_folder,
                        skip_face_gen=False):
        self.__make_debug_out_folder(debug_out_folder)

        self.__start_stage(len(filenames))
        for f in filenames:
            if self.__step_stage():
                break
            try:
                ext = tools.get_low_ext(f)
                if ext in tools.IMAGE_EXTS:
                    encoded_faces, media = self.recognize_image(f)
                elif ext in tools.VIDEO_EXTS:
                    encoded_faces, media = self.recognize_video(f)
                else:
                    logging.warning(f'Unknown ext: {ext}')
                    continue
                if media is None:
                    continue
                self.__db.insert(f, encoded_faces, commit=False)
                if debug_out_folder:
                    debug_out_file_name = self.__extract_filename(f)
                    self.__save_debug_images(
                        encoded_faces, media,
                        debug_out_folder, debug_out_file_name,
                        is_video=ext in tools.VIDEO_EXTS,
                        skip_face_gen=skip_face_gen)
            except Exception as ex:
                logging.exception(f'Image {f} recognition failed')
        self.__end_stage()

    def reencode_files(self, files_faces):
        for ff in files_faces:
            try:
                encoded_faces = ff['faces']
                filename = ff['filename']
                ext = tools.get_low_ext(filename)
                if ext in tools.IMAGE_EXTS:
                    self.reencode_image(filename, encoded_faces)
                    encoded_faces = self.__filter_encoded_faces(encoded_faces)
                elif ext in tools.VIDEO_EXTS:
                    encoded_faces, media = self.recognize_video(filename)
                else:
                    logging.warning(f'Unknown ext: {ext}')
                    continue
                self.__db.insert(filename, encoded_faces, commit=False)
            except Exception as ex:
                logging.exception(f'{filename} reencoding failed')
        self.__end_stage()

    def __get_files_faces_by_filter(self, fltr):
        logging.debug(f'Get by filter: {fltr}')
        tp = fltr['type']
        if tp == 'unmatched':
            return self.__db.get_unmatched()
        elif tp == 'all':
            return self.__db.get_all()
        elif tp == 'weak':
            return self.__db.get_weak(fltr['path'])
        elif tp == 'weak_unmatched':
            return self.__db.get_weak_unmatched(fltr['path'])
        elif tp == 'folder':
            return self.__db.get_folder(fltr['path'])
        elif tp == 'name':
            return self.__db.get_by_name(fltr['path'], fltr['name'])
        else:
            raise Exception(f'Unknown filter type: {tp}')

    def clusterize(self, fltr, debug_out_folder):
        if self.__init_stage('clusterize', locals()):
            return
        count, files_faces = self.__get_files_faces_by_filter(fltr)
        files_faces = list(tools.filter_images(files_faces))
        self.__clusterize(files_faces, debug_out_folder)

    def match(self, fltr, debug_out_folder, save_all_faces,
              skip_face_gen=False):
        if self.__init_stage('match', locals()):
            return
        count, files_faces = self.__get_files_faces_by_filter(fltr)
        self.__start_stage(count)
        self.__match_files_faces(files_faces,
                                 debug_out_folder,
                                 save_all_faces,
                                 skip_face_gen)

    def save_faces(self, fltr, debug_out_folder):
        if self.__init_stage('save_faces', locals()):
            return
        count, files_faces = self.__get_files_faces_by_filter(fltr)
        self.__start_stage(count)
        self.__save_faces(files_faces, debug_out_folder)

    def __match_files_faces(
            self, files_faces, debug_out_folder,
            save_all_faces=False, skip_face_gen=False):
        cnt_all = 0
        cnt_changed = 0
        for ff in files_faces:
            if self.__step_stage():
                break
            filename = ff['filename']
            logging.info(f"match image: {filename}")
            is_video = tools.get_low_ext(filename) in tools.VIDEO_EXTS
            if not self.__match_faces(ff['faces']):
                continue
            for face in ff['faces']:
                cnt_all += 1
                changed = False
                if 'oldname' in face and face['oldname'] != face['name']:
                    self.__db.set_name(face['face_id'], face['name'],
                                       face['dist'], face['pattern'],
                                       commit=False)
                    cnt_changed += 1
                    changed = True
                    logging.info(
                        f"face {face['face_id']} in file '{ff['filename']}' " +
                        f"changed '{face['oldname']}' -> '{face['name']}'")
                if debug_out_folder and (changed or save_all_faces):
                    media = tools.load_media(filename,
                                             self.__max_size,
                                             self.__max_video_frames,
                                             self.__video_frames_step)
                    debug_out_file_name = self.__extract_filename(filename)
                    self.__save_debug_images(
                        (face,), media,
                        debug_out_folder, debug_out_file_name,
                        is_video=is_video,
                        skip_face_gen=skip_face_gen)
        self.__end_stage()
        logging.info(f'match done: count: {cnt_all}, changed: {cnt_changed}')

    def __save_faces(self, files_faces, debug_out_folder):
        for ff in files_faces:
            if self.__step_stage():
                break
            filename = ff['filename']
            logging.info(f"save faces from image: {filename}")
            media = tools.load_media(filename,
                                     self.__max_size,
                                     self.__max_video_frames,
                                     self.__video_frames_step)
            debug_out_file_name = self.__extract_filename(filename)
            is_video = tools.get_low_ext(filename) in tools.VIDEO_EXTS
            self.__save_debug_images(
                ff['faces'], media,
                debug_out_folder, debug_out_file_name, is_video=is_video)
        self.__end_stage()

    def __filter_encoded_faces(self, encoded_faces):
        res = []
        for enc in encoded_faces:
            if 'profile_angle' in enc and \
                    enc['profile_angle'] > self.__max_face_profile_angle:
                logging.debug(f"Skip profile face: {enc['profile_angle']}")
                continue
            if not tools.test_landmarks(enc['landmarks']):
                logging.debug(f'Skip incorrect landmark')
                continue
            res.append(enc)
            logging.debug(f"profile face: {enc['profile_angle']}")
        return res

    def recognize_folder(self, folder, debug_out_folder,
                         reencode=False, skip_face_gen=False):
        if self.__init_stage('recognize_folder', locals()):
            return
        filenames = self.__get_media_from_folder(folder)

        if not reencode:
            filenames = list(set(filenames) - set(self.__db.get_files(folder)))
            filenames.sort()

        self.recognize_files(filenames, debug_out_folder, skip_face_gen)

    def remove_folder(self, folder):
        if self.__init_stage('remove_folder', locals()):
            return
        files_faces = self.__db.get_folder(folder)
        for ff in files_faces:
            logging.info(f"remove from DB: {ff['filename']}")
            self.__db.remove(ff['filename'], False)
            if self.__cdb is not None:
                for face in ff['faces']:
                    self.__cdb.remove_face(face['face_id'])
        # delete files without faces
        files = self.__db.get_files(folder)
        for f in files:
            logging.info(f"remove image: {f}")
            self.__db.remove(f, False)
        self.__end_stage()

    def __get_media_from_folder(self, folder):
        files = tools.list_files(
            folder,
            tools.IMAGE_EXTS + tools.VIDEO_EXTS,
            self.__nomedia_files)
        files.sort()
        return files

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
            self, encoded_faces, media, debug_out_folder, debug_out_file_name,
            is_video=False, skip_face_gen=False):

        if is_video:
            encoded_faces = tools.reduce_faces_from_video(
                encoded_faces, self.__min_video_face_count)

        for enc in encoded_faces:
            name = enc['name']
            if name == '':
                name = 'unknown_00000'

            out_folder = os.path.join(debug_out_folder, name)

            top, right, bottom, left = enc['box']

            prefix = '{}_{:03d}'.format(name, int(enc['dist'] * 100))
            out_filename = os.path.join(
                out_folder,
                f'{prefix}_{debug_out_file_name}_{left}x{top}.jpg')

            if self.__cdb is not None:
                if not self.__cdb.check_face(enc['face_id']):
                    out_stream = io.BytesIO()
                    tools.save_face(out_stream, media.get(enc['frame']), enc,
                                    self.__debug_out_image_size,
                                    media.filename())
                    self.__cdb.save_face(enc['face_id'],
                                         out_stream.getvalue())
                    logging.debug(f"face {enc['face_id']} cached")
                if not skip_face_gen:
                    self.__cdb.add_to_cache(enc['face_id'], out_filename)
            elif not skip_face_gen:
                self.__make_debug_out_folder(out_folder)
                tools.save_face(out_filename, media.get(enc['frame']), enc,
                                self.__debug_out_image_size,
                                media.filename())
                logging.debug(f'face saved to: {out_filename}')

    def get_faces_by_face(self, filename, debug_out_folder,
                          remove_file=False):
        if self.__init_stage('get_faces_by_face', locals()):
            return

        image = tools.LazyImage(filename, self.__max_size)

        encoded_faces = self.encode_faces(image.get())
        face = encoded_faces[0]
        logging.debug(f'found face: {face}')

        all_encodings = self.__db.get_all_encodings(self.__max_workers)

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

        self.__start_stage(len(filtered))
        for dist, info in filtered:
            if self.__step_stage():
                break
            fname, face = info
            face['dist'] = dist
            media = tools.load_media(fname,
                                     self.__max_size,
                                     self.__max_video_frames,
                                     self.__video_frames_step)
            debug_out_file_name = self.__extract_filename(fname)
            self.__save_debug_images(
                (face,), media,
                debug_out_folder, debug_out_file_name)
        if remove_file:
            logging.debug(f'removing temp file: {filename}')
            os.remove(filename)

    def stop(self, save=False):
        logging.info(f'Stop called ({save})')
        self.__status['stop'] = True
        self.__status['save'] = save

    def __init_stage(self, cmd, args):
        self.__status['state'] = cmd
        self.__status['args'] = {}
        del args['self']
        for k, v in args.items():
            args[k] = str(v)
        self.__status['args'] = args
        return self.__status['stop']

    def __start_stage(self, count):
        logging.info(
            f'Stage {self.__status["state"]} for {count} steps started')
        self.__status['count'] = count
        self.__status['current'] = 0
        self.__status['starttime'] = time.time()

    def __step_stage(self, step=1):
        self.__status['current'] += step
        return self.__status['stop']

    def __end_stage(self):
        if self.__status.get('save', True):
            logging.info(f'Commit transaction')
            if self.__db is not None:
                self.__db.commit()
            if self.__cdb is not None:
                self.__cdb.commit()
        else:
            logging.info(f'Rollback transaction')
            if self.__db is not None:
                self.__db.rollback()
            if self.__cdb is not None:
                self.__cdb.rollback()
        self.__status['stop'] = False


def createRecognizer(patt, cfg, cdb=None, db=None, status=None):
    return Recognizer(patt,
                      model=cfg['recognition']['model'],
                      num_jitters=cfg['recognition']['num_jitters'],
                      threshold=cfg['recognition']['threshold'],
                      threshold_weak=cfg['recognition']['threshold_weak'],
                      threshold_clusterize=cfg['recognition'][
                          'threshold_clusterize'],
                      threshold_equal=cfg['recognition']['threshold_equal'],
                      max_image_size=cfg['processing']['max_image_size'],
                      max_video_frames=cfg['processing']['max_video_frames'],
                      video_frames_step=cfg['processing']['video_frames_step'],
                      min_face_size=cfg['recognition']['min_face_size'],
                      max_face_profile_angle=cfg['recognition'][
                          'max_face_profile_angle'],
                      min_video_face_count=cfg['recognition'][
                          'min_video_face_count'],
                      debug_out_image_size=cfg['processing'][
                          'debug_out_image_size'],
                      encoding_model=cfg['recognition']['encoding_model'],
                      distance_metric=cfg['recognition']['distance_metric'],
                      max_workers=cfg['processing']['max_workers'],
                      video_batch_size=cfg['processing']['video_batch_size'],
                      nomedia_files=cfg['files']['nomedia_files'].split(':'),
                      cdb=cdb,
                      db=db,
                      status=status)


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

    patt = patterns.createPatterns(cfg)
    patt.load()

    cachedb_file = cfg['files']['cachedb']
    if cachedb_file:
        cdb = cachedb.CacheDB(cachedb_file)
    else:
        cdb = None

    tools.cuda_init()
    db = recdb.RecDB(cfg['files']['db'], args.dry_run)
    rec = createRecognizer(patt, cfg, cdb, db)

    signal.signal(signal.SIGINT, lambda sig, frame: rec.stop())

    if args.action == 'recognize_image':
        print(rec.recognize_image(args.input)[0])
    elif args.action == 'recognize_video':
        print(rec.calc_names_in_video(rec.recognize_video(args.input)[0]))
    elif args.action == 'recognize_folder':
        rec.recognize_folder(args.input, args.output, args.reencode)
    elif args.action == 'remove_folder':
        rec.remove_folder(args.input)
    elif args.action == 'match_unmatched':
        rec.match({'type': 'unmatched'}, args.output, False)
    elif args.action == 'match_all':
        rec.match({'type': 'all'}, args.output, False)
    elif args.action == 'match_folder':
        rec.match({'type': 'folder', 'path': args.input}, args.output, True)
    elif args.action == 'clusterize_unmatched':
        rec.clusterize({'type': 'unmatched'}, args.output)
    elif args.action == 'save_faces':
        rec.save_faces({'type': 'folder', 'path': args.input}, args.output)
    elif args.action == 'get_faces_by_face':
        rec.get_faces_by_face(args.input, args.output)


if __name__ == '__main__':
    main()
