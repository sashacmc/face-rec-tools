#!/usr/bin/python3

import os
import cv2
import sys
import logging
import argparse
import collections
import face_recognition

import log
import recdb
import tools
import patterns


class Recognizer(object):
    def __init__(self, patterns, model='hog', num_jitters=1, threshold=0.5):
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

            if 'name' in encoded_faces[i] and encoded_faces[i]['name']:
                encoded_faces[i]['oldname'] = encoded_faces[i]['name']
            encoded_faces[i]['name'] = name

    def clusterize(self, files_faces, debug_out_folder=None):
        encs = []
        for filename, faces in files_faces:
            encs.append([dlib.vector(enc) for e['encoding'] in faces])

        labels = dlib.chinese_whispers_clustering(encs, 0.4)

        for i in range(len(files_faces)):
            for j in range(len(files_faces[i][1])):
                if files_faces[i]['faces'][j]['name'] != '':
                    continue

                files_faces[i]['faces'][j]['name'] = f'unknown_{labels[i]}'

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
        for filename, faces in files_faces:
            self.match(faces)
            for face in faces:
                cnt_all += 1
                if face['name']:
                    db.set_name(face['face_id'], face['name'])
                    cnt_changed += 1
                    logging.info(
                        f"face {face['face_id']} in file '{filename}' " +
                        f"matched to '{face['name']}'")
        logging.info(f'match_unmatched: {cnt_all}, changed: {cnt_changed}')

    def match_all(self, db, debug_out_folder):
        files_faces = db.get_all()
        cnt_all = 0
        cnt_changed = 0
        for filename, faces in files_faces:
            self.match(faces)
            for face in faces:
                cnt_all += 1
                if 'oldname' in face and face['oldname'] != face['name']:
                    db.set_name(face['face_id'], face['name'])
                    cnt_changed += 1
                    logging.info(
                        f"face {face['face_id']} in file '{filename}' " +
                        f"chnaged '{face['name']}' -> '{face['oldname']}'")
        logging.info(f'match_all: {cnt_all}, changed: {cnt_changed}')

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
            d = (bottom - top) // 2
            out_image = image[
                max(0, top - d):bottom + d,
                max(0, left - d):right + d]
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

            out_filename = os.path.join(
                out_folder, f'{debug_out_file_name}_{i}_{name}.jpg')

            cv2.imwrite(out_filename, out_image)
            logging.debug(f'face saved to: {out_filename}')


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['recognize_image',
                 'recognize_folder',
                 'match_unmatched',
                 'match_all'])
    parser.add_argument('-p', '--patterns', help='Patterns file')
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-i', '--input', help='Input file or folder')
    parser.add_argument('-o', '--output', help='Output folder for faces')
    parser.add_argument('-d', '--dry-run', help='Do''t modify DB',
                        action='store_true')
    return parser.parse_args()


def main():
    import sys

    args = args_parse()
    log.initLogger(args.logfile)

    patt = patterns.Patterns(args.patterns)

    rec = Recognizer(patt, 'cnn')
    db = recdb.RecDB('rec.db', args.dry_run)

    if args.action == 'recognize_image':
        print(rec.recognize_image(args.input, args.output))
    elif args.action == 'recognize_folder':
        patt.load()
        rec.recognize_folder(args.input, db)
    elif args.action == 'match_unmatched':
        patt.load()
        rec.match_unmatched(db, args.output)
    elif args.action == 'match_all':
        patt.load()
        rec.match_all(db, args.output)


if __name__ == '__main__':
    main()
