#!/usr/bin/python3

import os
import time
import atexit
import sqlite3
import argparse
import threading

SCHEMA = '''
CREATE TABLE IF NOT EXISTS face_images (
    "face_id" INTEGER PRIMARY KEY NOT NULL,
    "data" BLOB
);

CREATE TABLE IF NOT EXISTS cache (
    "filename" TEXT,
    "face_id" INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS cache_filename ON cache (filename);
'''

AUTOCOMMIT_TIMEOUT = 60  # sec


class CacheDB(object):
    def __init__(self, filename):
        self.__conn = sqlite3.connect(
            filename,
            check_same_thread=False)

        self.__conn.execute('pragma journal_mode=off;')
        self.__conn.executescript(SCHEMA)
        self.__autocommit_time = time.time()
        self.__lock = threading.RLock()
        atexit.register(self.commit)

    def __del__(self):
        self.commit()

    def commit(self):
        with self.__lock:
            self.__conn.commit()
            self.__autocommit_time = time.time()

    def __autocommit(self):
        tm = time.time()
        if self.__autocommit_time + AUTOCOMMIT_TIMEOUT < tm:
            self.__conn.commit()
            self.__autocommit_time = tm

    def save_face(self, face_id, data):
        with self.__lock:
            c = self.__conn.cursor()
            c.execute(
                'INSERT INTO face_images (face_id, data) \
                 VALUES (?, ?)', (face_id, data))
            self.__autocommit()

    def check_face(self, face_id):
        with self.__lock:
            c = self.__conn.cursor()
            res = c.execute('SELECT face_id FROM face_images WHERE face_id=?',
                            (face_id,))
            return res.fetchone() is not None

    def remove_face(self, face_id):
        with self.__lock:
            c = self.__conn.cursor()
            c.execute('DELETE FROM face_images WHERE face_id=?', (face_id,))
            self.__autocommit()

    def list_cache(self):
        with self.__lock:
            self.commit()
            c = self.__conn.cursor()
            res = c.execute('SELECT filename FROM cache')
            return [r[0] for r in res.fetchall()]

    def clean_cache(self):
        with self.__lock:
            self.commit()
            c = self.__conn.cursor()
            c.execute('DELETE FROM cache')
            self.commit()

    def add_to_cache(self, face_id, filename):
        with self.__lock:
            c = self.__conn.cursor()
            c.execute(
                'INSERT OR REPLACE INTO cache (face_id, filename) \
                 VALUES (?, ?)', (face_id, filename))
            self.__autocommit()

    def get_from_cache(self, filename):
        with self.__lock:
            c = self.__conn.cursor()
            res = c.execute('SELECT data FROM face_images \
                             JOIN cache ON face_images.face_id=cache.face_id \
                             WHERE filename=?', (filename,))
            row = res.fetchone()
            if row is not None:
                return row[0]
            else:
                return None

    def save_from_cache(self, filename, out_filename):
        with self.__lock:
            data = self.get_from_cache(filename)
            os.makedirs(os.path.split(filename)[0], exist_ok=True)
            with open(out_filename, 'wb') as f:
                f.write(data)

    def remove_from_cache(self, filename):
        with self.__lock:
            c = self.__conn.cursor()
            c.execute('DELETE FROM cache WHERE filename=?', (filename,))
            self.__autocommit()


def __speed_test(db):
    data = 'z' * 4000
    count = 1000
    for i in range(count):
        db.save_face(i, data)
    del db


def speed_test(db):
    import cProfile

    cProfile.runctx('__speed_test(db)',
                    {'__speed_test': __speed_test, 'db': db}, {})


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['clean_cache',
                 'list_cache',
                 'save_file',
                 'remove_from_cache',
                 'speed_test'])
    parser.add_argument('-d', '--database', help='Database file')
    parser.add_argument('-f', '--file', help='File name')
    parser.add_argument('-o', '--out-file', help='Out file name')
    return parser.parse_args()


def main():
    args = args_parse()
    db = CacheDB(args.database)

    if args.action == 'clean_cache':
        db.clean_cache()
    if args.action == 'list_cache':
        for f in db.list_cache():
            print(f)
    elif args.action == 'save_file':
        db.save_from_cache(args.file, args.out_file)
    elif args.action == 'remove_from_cache':
        db.remove_from_cache(args.file)
    elif args.action == 'speed_test':
        speed_test(db)


if __name__ == '__main__':
    main()
