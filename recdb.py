#!/usr/bin/python3

import io
import numpy
import sqlite3

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
    "name" TEXT
);

CREATE TRIGGER IF NOT EXISTS faces_before_delete
BEFORE DELETE ON images
BEGIN
    DELETE FROM faces WHERE image_id=OLD.id;
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
            filename, detect_types=sqlite3.PARSE_DECLTYPES)
        self.__conn.executescript(SCHEMA)
        self.__readonly = readonly

    def insert(self, filename, rec_result):
        # rec_result =
        #   [{'box': (l, b, r, t),
        #     'encoding': BLOB,
        #     'name': name
        #    }, ...]
        if self.__readonly:
            return []

        c = self.__conn.cursor()

        c.execute('DELETE FROM images WHERE filename=?', (filename,))

        image_id = c.execute(
            'INSERT INTO images (filename) \
             VALUES (?)', (filename,)).lastrowid

        res = []
        for face in rec_result:
            face_id = c.execute(
                'INSERT INTO faces (image_id, box, encoding, name) \
                 VALUES (?, ?, ?, ?)',
                (image_id, str(face["box"]), face['encoding'], face['name'])
            ).lastrowid

            res.append(face_id)

        self.__conn.commit()

        return res

    def get_faces(self):
        c = self.__conn.cursor()
        res = c.execute('SELECT image_id, box, encoding FROM faces')

        return [{'id': r[0], 'box': r[1], 'encoding': [2]}
                for r in res.fetchall()]

    def set_name(self, face_id, name):
        if self.__readonly:
            return
        c = self.__conn.cursor()
        c.execute('UPDATE faces SET name=? WHERE id=?', (name, face_id))
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

    def get_unmatched(self):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT filename, faces.id, box, encoding, name \
             FROM images JOIN faces ON images.id=faces.image_id \
             WHERE name=""')

        return self.__build_files_faces(res.fetchall())

    def get_all(self):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT filename, faces.id, box, encoding, name \
             FROM images JOIN faces ON images.id=faces.image_id')

        return self.__build_files_faces(res.fetchall())

    def __build_files_faces(self, res):
        files_faces = []
        filename = ''
        faces = []

        for r in res:
            if r[0] != filename:
                if filename != '':
                    files_faces.append((filename, faces))
                filename = r[0]
                faces = []
            faces.append({
                'face_id': r[1],
                'box': r[2],
                'encoding': r[3],
                'name': r[4]})

        if filename != '':
            files_faces.append((filename, faces))

        return files_faces

    def mark_as_synced(self, filename):
        if self.__readonly:
            return
        pass


if __name__ == '__main__':
    db = RecDB('tt.db')
