#!/usr/bin/python3

import os
import sys
import logging
import argparse

sys.path.insert(0, os.path.abspath('..'))

from face_rec_tools import log  # noqa
from face_rec_tools import recdb  # noqa
from face_rec_tools import tools  # noqa
from face_rec_tools import plexdb  # noqa
from face_rec_tools import config  # noqa
from face_rec_tools import cachedb  # noqa
from face_rec_tools import patterns  # noqa
from face_rec_tools import recognizer  # noqa


TAG_PREFIX = 'person:'


class PlexSync(object):
    def __init__(self, names, recdb, plexdb, min_video_face_count):
        self.__names = names
        self.__recdb = recdb
        self.__plexdb = plexdb
        self.__min_video_face_count = int(min_video_face_count)

    def __create_tags(self):
        for name in self.__names:
            tag = TAG_PREFIX + name
            if self.__plexdb.tag_exists(tag):
                continue
            self.__plexdb.create_tag(tag, commit=False)
            logging.info(f'Tag "{tag}" added to plex db')
        self.__plexdb.commit()

    def set_tags(self, resync=False):
        logging.info(f'Set tags started ({resync})')
        self.__create_tags()

        if resync:
            files_faces = self.__recdb.get_all()
        else:
            files_faces = self.__recdb.get_unsynced()

        images_count = 0
        faces_count = 0
        for ff in tools.reduce_faces_from_videos(files_faces,
                                                 self.__min_video_face_count):
            filename = ff['filename']
            self.__plexdb.clean_tags(filename, tag_prefix=TAG_PREFIX,
                                     commit=False)
            tags = []
            for face in ff['faces']:
                name = face['name']
                if name in self.__names:
                    tags.append(TAG_PREFIX + name)

            logging.debug(f"sync tags for image: {filename}: " + str(tags))
            if len(tags) != 0:
                self.__plexdb.set_tags(filename, tags, commit=False)
            self.__recdb.mark_as_synced(filename, commit=False)
            images_count += 1
            faces_count += len(tags)
        self.__plexdb.commit()
        self.__recdb.commit()

        logging.info(
            f'Set tags done: images={images_count} faces={faces_count}')

    def remove_tags(self):
        logging.info(f'Remove tags started')
        self.__plexdb.delete_tags(TAG_PREFIX, cleanup=True)
        logging.info(f'Remove tags done')

    def sync_new(self, cfg, patt, folders, exts):
        logging.info(f'Sync new started')
        cachedb_file = cfg['main']['cachedb']
        cache_path = cfg['server']['face_cache_path']
        if cachedb_file:
            cdb = cachedb.CacheDB(cachedb_file)
        else:
            cdb = None
        rec = recognizer.createRecognizer(patt, cfg, cdb, self.__recdb)

        for folder in folders:
            plex_files = self.__plexdb.get_files(folder)
            plex_files = set(
                filter(lambda f: os.path.splitext(f)[1].lower() in exts,
                       plex_files))
            rec_files = set(self.__recdb.get_files(folder))
            filenames = sorted(plex_files - rec_files)
            count = len(filenames)
            if count != 0:
                logging.info(f'Adding {count} files from {folder}')
                rec.recognize_files(filenames, cache_path)
            else:
                logging.info(f'No files to add from {folder}')
        self.set_tags()

    def sync_deleted(self, folders):
        logging.info(f'Sync deleted started')
        for folder in folders:
            logging.info(f'Check {folder}')
            plex_files = set(self.__plexdb.get_files(folder))
            rec_files = set(self.__recdb.get_files(folder))
            filenames = sorted(rec_files - plex_files)
            if len(filenames) != 0:
                for filename in filenames:
                    logging.info(f'removing from recdb: {filename}')
                    self.__recdb.remove(filename, commit=False)
            else:
                logging.info(f'No files to remove from {folder}')
            self.__recdb.commit()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['set_tags',
                 'remove_tags',
                 'sync_new',
                 'sync_deleted'])
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('-r', '--resync', help='Resync all',
                        action='store_true')
    parser.add_argument('-d', '--dry-run', help='Do''t modify DB',
                        action='store_true')
    return parser.parse_args()


def main():
    args = args_parse()
    cfg = config.Config(args.config)

    log.initLogger(args.logfile)
    logging.basicConfig(level=logging.DEBUG)

    patt = patterns.createPatterns(cfg)
    patt.load()
    names = set([p['name'] for p in patt.persons()])
    names.remove('trash')

    rdb = recdb.RecDB(cfg['main']['db'], args.dry_run)
    pdb = plexdb.PlexDB(cfg['plex']['db'], args.dry_run)

    pls = PlexSync(names, rdb, pdb,
                   min_video_face_count=cfg['main']['min_video_face_count'])

    if args.action == 'set_tags':
        pls.set_tags(resync=args.resync)
    elif args.action == 'remove_tags':
        pls.remove_tags()
    elif args.action == 'sync_new':
        folders = cfg['plex']['folders'].split(':')
        pls.sync_new(cfg, patt, folders, ('.jpg',))
    elif args.action == 'sync_deleted':
        folders = cfg['plex']['folders'].split(':')
        pls.sync_deleted(folders)


if __name__ == '__main__':
    main()
