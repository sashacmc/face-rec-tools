#!/usr/bin/python3

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
    "encoding" BlOB,
    "name" TEXT
);

CREATE TRIGGER IF NOT EXISTS faces_before_delete
BEFORE DELETE ON images
BEGIN
    DELETE FROM faces WHERE image_id=OLD.id;
END;
'''


class RecDB(object):
    def __init__(self, filename):
        self.__conn = sqlite3.connect(filename)
        self.__conn.executescript(SCHEMA)

    def insert(self, filename, rec_result):
        # rec_result =
        #   [{'box': (l, b, r, t),
        #     'encoding': BLOB,
        #     'name': name
        #    }, ...]

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

    def mark_as_synced(self, filename):
        pass


if __name__ == '__main__':
    db = RecDB('tt.db')
