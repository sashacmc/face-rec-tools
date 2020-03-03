#!/usr/bin/python3

import re
import os
import logging
import argparse

import log
import recdb
import tools
import config
import patterns
import recognizer
import faceencoder


def get_from_db(files_faces, db, filename):
    fname = os.path.split(filename)[1]
    try:
        fname, fleft, ftop = re.search('(.*)_(\d+)x(\d+)\.jpg', fname).groups()
        if fname[:2] == '0_':
            fname = fname[2:]
    except AttributeError:
        return None, None
    except IndexError:
        return None, None

    fleft = int(fleft)
    ftop = int(ftop)

    for ff in files_faces:
        if fname in ff['filename']:
            for face in ff['faces']:
                top, right, bottom, left = face['box']
                if ftop == top and fleft == left:
                    return ff['filename'], face['box']

    return None, None


def update(patt, db, num_jitters, encoding_model, max_size, out_size):
    encoder = faceencoder.FaceEncoder(
        encoding_model=encoding_model,
        num_jitters=num_jitters,
        align=True)

    files_faces = db.get_all()

    encodings, names, filenames = patt.encodings()
    for patt_fname, enc in zip(filenames, encodings):
        fname, box = get_from_db(files_faces, db, patt_fname)
        if fname is None:
            logging.warning(f'Not found in db: {patt_fname}')
            continue

        logging.debug(f'Found in db file: {fname} {box}')

        try:
            image = tools.read_image(fname, max_size)
        except Exception as ex:
            logging.warning(f'Cant''t read image: {fname}: ' + str(ex))
            continue

        try:
            encodings, landmarks = encoder.encode(image, (box,))

            enc = {'box': box,
                   'encoding': encodings[0],
                   'frame': 0,
                   'landmarks': landmarks[0]}
            tools.save_face(patt_fname,
                            image, enc,
                            out_size,
                            fname)
            logging.info(f'Updated: {patt_fname}')
        except Exception as ex:
            logging.exception(f'Failed: {patt_fname}')


def update_db(db, rec):
    files_faces = db.get_all()
    logging.info(f'Start reencode {len(files_faces)} files')
    rec.reeencode_files(files_faces, db, None)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('files', nargs='*', help='Files with one face')
    return parser.parse_args()


def main():
    args = args_parse()
    cfg = config.Config(args.config)

    log.initLogger(args.logfile)
    logging.basicConfig(level=logging.DEBUG)

    db = recdb.RecDB(cfg['main']['db'])
    patt = patterns.Patterns(cfg['main']['patterns'],
                             cfg['main']['model'])
    patt.load()

    update(patt, db,
           int(cfg['main']['num_jitters']),
           cfg['main']['encoding_model'],
           int(cfg['main']['max_image_size']),
           int(cfg['main']['debug_out_image_size']))

    rec = recognizer.createRecognizer(patt, cfg)

    update_db(db, rec)


if __name__ == '__main__':
    main()
