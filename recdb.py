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
    "name" TEXT
);

CREATE TABLE IF NOT EXISTS persons (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "face_id" INTEGER,
    "name" TEXT,
    "delta" FLOAT
);

CREATE TRIGGER IF NOT EXISTS faces_before_delete
BEFORE DELETE ON images
BEGIN
    DELETE FROM faces WHERE image_id=OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS persons_before_delete
BEFORE DELETE ON faces
BEGIN
    DELETE FROM persons WHERE face_id=OLD.id;
END;
'''


class RecDB(object):
    def __init__(self, filename):
        self.__conn = sqlite3.connect(filename)
        self.__conn.executescript(SCHEMA)

    def insert(self, filename, rec_result):
        # rec_result = [{'box': (l, b, r, t), 'names': [(delta, 'name'), ]}, ]
        c = self.__conn.cursor()

        c.execute('DELETE FROM images WHERE filename=?', (filename,))

        image_id = c.execute(
            'INSERT INTO images (filename) \
             VALUES (?)', (filename,)).lastrowid

        for face in rec_result:
            names = face['names']
            if len(names):
                name = names[0][1]
            else:
                name = ''

            face_id = c.execute(
                'INSERT INTO faces (image_id, box, name) \
                 VALUES (?, ?, ?)',
                (image_id, str(face["box"]), name)).lastrowid

            for person in names:
                c.execute(
                    'INSERT INTO persons (face_id, name, delta) \
                     VALUES (?, ?, ?)',
                    (face_id, person[1], person[0]))

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
            'SELECT faces.id, faces.box \
             FROM images JOIN faces ON images.id=faces.image_id \
             WHERE filename=?', (filename,))

        print(f'File: {filename}')
        for r in res.fetchall():
            print(f'\tBox: {r[1]}')
            res = c.execute(
                'SELECT name, delta \
                 FROM persons \
                 WHERE face_id=?', (r[0],))
            for r in res.fetchall():
                print(f'\t\t{r[0]}: {r[1]}')

    def mark_as_synced(self, filename):
        pass


if __name__ == '__main__':
    db = RecDB('tt.db')
