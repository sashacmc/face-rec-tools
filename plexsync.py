#!/usr/bin/python3

import logging
import argparse

import log
import recdb
import plexdb
import config
import patterns


TAG_PREFIX = 'person:'


class PlexSync(object):
    def __init__(self, names, recdb, plexdb):
        self.__names = names
        self.__recdb = recdb
        self.__plexdb = plexdb

    def __create_tags(self):
        for name in self.__names:
            tag = TAG_PREFIX + name
            if self.__plexdb.tag_exists(tag):
                continue
            self.__plexdb.create_tag(tag)
            logging.info(f'Tag "{tag}" added to plex db')

    def set_tags(self, resync=False):
        logging.info(f'Set tags started ({resync})')
        self.__create_tags()

        if resync:
            files_faces = self.__recdb.get_all()
        else:
            files_faces = self.__recdb.get_unsynced()

        images_count = 0
        faces_count = 0
        for ff in files_faces:
            filename = ff['filename']
            self.__plexdb.clean_tags(filename, tag_prefix=TAG_PREFIX)
            tags = []
            for face in ff['faces']:
                name = face['name']
                if name in self.__names:
                    tags.append(TAG_PREFIX + name)

            logging.debug(f"sync tags for image: {filename}: " + str(tags))
            if len(tags) != 0:
                self.__plexdb.set_tags(filename, tags)
            self.__recdb.mark_as_synced(filename, commit=False)
            images_count += 1
            faces_count += len(tags)
        self.__recdb.commit()

        logging.info(
            f'Set tags done: images={images_count} faces={faces_count}')

    def remove_tags(self):
        logging.info(f'Remove tags started')
        self.__plexdb.delete_tags(TAG_PREFIX, cleanup=True)
        logging.info(f'Remove tags done')


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['set_tags',
                 'remove_tags'])
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('-r', '--resync', help='Resync all',
                        action='store_true')
    return parser.parse_args()


def main():
    args = args_parse()
    cfg = config.Config(args.config)

    log.initLogger(args.logfile)
    logging.basicConfig(level=logging.DEBUG)

    patt = patterns.Patterns(cfg['main']['patterns'],
                             cfg['main']['model'])
    patt.load()
    names = set(patt.names())
    names.remove('trash')

    rdb = recdb.RecDB(cfg['main']['db'])
    pdb = plexdb.PlexDB(cfg['plex']['db'])

    pls = PlexSync(names, rdb, pdb)

    if args.action == 'set_tags':
        pls.set_tags(resync=args.resync)
    elif args.action == 'remove_tags':
        pls.remove_tags()


if __name__ == '__main__':
    main()
