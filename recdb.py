#!/usr/bin/python3

import io
import os
import json
import numpy
import sqlite3
import argparse

SCHEMA = '''
CREATE TABLE IF NOT EXISTS images (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "filename" TEXT,
    "synced" INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS faces (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "image_id" INTEGER,
    "box" TEXT,
    "encoding" array,
    "name" TEXT,
    "dist" FLOAT
);

CREATE TRIGGER IF NOT EXISTS faces_before_delete
BEFORE DELETE ON images
BEGIN
    DELETE FROM faces WHERE image_id=OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS set_images_unsync
AFTER UPDATE ON faces
BEGIN
    UPDATE images SET synced=0 WHERE id=OLD.image_id;
END;
'''


def adapt_array(arr):
    out = io.BytesIO()
    numpy.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return numpy.load(out)


class RecDB(object):
    def __init__(self, filename, readonly=False):
        sqlite3.register_adapter(numpy.ndarray, adapt_array)
        sqlite3.register_converter('array', convert_array)

        self.__conn = sqlite3.connect(
            filename,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False)

        self.__conn.executescript(SCHEMA)
        self.__readonly = readonly

    def insert(self, filename, rec_result):
        # rec_result =
        #   [{'box': (l, b, r, t),
        #     'encoding': BLOB,
        #     'name': name
        #     'dist': dist
        #    }, ...]
        if self.__readonly:
            return []

        c = self.__conn.cursor()

        c.execute('DELETE FROM images WHERE filename=?', (filename,))

        image_id = c.execute(
            'INSERT INTO images (filename) \
             VALUES (?)', (filename,)).lastrowid

        res = []
        for i, face in enumerate(rec_result):
            rec_result[i]['face_id'] = c.execute(
                'INSERT INTO faces (image_id, box, encoding, name, dist) \
                 VALUES (?, ?, ?, ?, ?)',
                (image_id,
                 json.dumps(face["box"]),
                 face['encoding'],
                 face['name'],
                 face['dist'])
            ).lastrowid

        self.__conn.commit()

    def get_all_faces(self):
        c = self.__conn.cursor()
        res = c.execute('SELECT image_id, box, encoding FROM faces')

        return [{'id': r[0], 'box': json.loads(r[1]), 'encoding': [2]}
                for r in res.fetchall()]

    def set_name(self, face_id, name, dist):
        if self.__readonly:
            return
        c = self.__conn.cursor()
        c.execute('UPDATE faces SET name=?, dist=? WHERE id=?',
                  (name, dist, face_id))
        self.__conn.commit()

    def get_names(self, filename):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT faces.name \
             FROM images JOIN faces ON images.id=faces.image_id \
             WHERE filename=?', (filename,))

        return [r[0] for r in res.fetchall()]

    def print_details(self, filename):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT faces.id, faces.box, faces.name \
             FROM images JOIN faces ON images.id=faces.image_id \
             WHERE filename=?', (filename,))

        print(f'File: {filename}')
        for r in res.fetchall():
            print(f'\tBox: {r[1]}')
            print(f'\tName: {r[2]}')

    def get_folders(self):
        c = self.__conn.cursor()
        res = c.execute('SELECT filename FROM images')

        fset = set()
        for r in res.fetchall():
            fset.add(os.path.split(r[0])[0])

        return list(fset)

    def get_files(self, folder=None):
        if folder is None:
            folder = ''
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT filename FROM images \
             WHERE filename LIKE ?', (folder + '%',))

        return [r[0] for r in res.fetchall()]

    def get_files_faces(self, where_clause, args=()):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT filename, faces.id, box, encoding, name, dist \
             FROM images JOIN faces ON images.id=faces.image_id ' +
            where_clause, args)
        return self.__build_files_faces(res.fetchall())

    def get_unmatched(self):
        return self.get_files_faces('WHERE name=""')

    def get_all(self):
        return self.get_files_faces('')

    def get_weak(self, folder):
        return self.get_files_faces(
            'WHERE filename LIKE ? AND name LIKE "%_weak"', (folder + '%',))

    def get_folder(self, folder):
        return self.get_files_faces('WHERE filename LIKE ?', (folder + '%',))

    def get_faces(self, filename):
        return self.get_files_faces('WHERE filename=?', (filename,))

    def get_unsynced(self):
        return self.get_files_faces('WHERE synced=0')

    def get_by_name(self, folder, name):
        return self.get_files_faces(
            'WHERE filename LIKE ? AND name=?', (folder + '%', name))

    def __build_files_faces(self, res):
        files_faces = []
        filename = ''
        faces = []

        for r in res:
            if r[0] != filename:
                if filename != '':
                    files_faces.append({'filename': filename, 'faces': faces})
                filename = r[0]
                faces = []
            faces.append({
                'face_id': r[1],
                'box': json.loads(r[2]),
                'encoding': r[3],
                'name': r[4],
                'dist': r[5]})

        if filename != '':
            files_faces.append({'filename': filename, 'faces': faces})

        return files_faces

    def mark_as_synced(self, filename):
        if self.__readonly:
            return
        c = self.__conn.cursor()
        c.execute('UPDATE images SET synced=1 WHERE filename=?', (filename,))
        self.__conn.commit()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['get_names',
                 'get_faces',
                 'print_details',
                 'get_folders',
                 'get_files'])
    parser.add_argument('-d', '--database', help='Database file')
    parser.add_argument('-f', '--file', help='File')
    return parser.parse_args()


def main():
    args = args_parse()
    db = RecDB(args.database)

    if args.action == 'get_names':
        print(db.get_names(args.file))
    elif args.action == 'get_faces':
        print(db.get_faces(args.file))
    elif args.action == 'print_details':
        db.print_details(args.file)
    elif args.action == 'get_folders':
        folders = db.get_folders()
        for f in folders:
            print(f)
    elif args.action == 'get_files':
        files = db.get_files(args.file)
        for f in files:
            print(f)


if __name__ == '__main__':
    main()
