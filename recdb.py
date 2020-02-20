#!/usr/bin/python3

import io
import os
import json
import numpy
import atexit
import logging
import sqlite3
import argparse

from imutils import paths

import log

SCHEMA = '''
CREATE TABLE IF NOT EXISTS files (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "filename" TEXT,
    "synced" INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS faces (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "file_id" INTEGER,
    "box" TEXT,
    "encoding" array,
    "landmarks" TEXT,
    "name" TEXT,
    "dist" FLOAT,
    "frame" INTEGER default 0
);

CREATE TRIGGER IF NOT EXISTS faces_before_delete
BEFORE DELETE ON files
BEGIN
    DELETE FROM faces WHERE file_id=OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS set_files_unsync
AFTER UPDATE ON faces
BEGIN
    UPDATE files SET synced=0 WHERE id=OLD.file_id;
END;

CREATE INDEX IF NOT EXISTS files_filename ON files (filename);
CREATE INDEX IF NOT EXISTS faces_file_id ON faces (file_id);
CREATE INDEX IF NOT EXISTS faces_name ON faces (name);
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
            'file:' + filename + ('?mode=ro' if readonly else ''),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
            uri=True)

        self.__conn.executescript(SCHEMA)
        self.__readonly = readonly
        atexit.register(self.commit)

    def commit(self):
        if self.__readonly:
            return
        self.__conn.commit()

    def insert(self, filename, rec_result, commit=True):
        # rec_result =
        #   [{'box': (l, b, r, t),
        #     'encoding': BLOB,
        #     'landmarks': {...},
        #     'name': name,
        #     'dist': dist,
        #     'frame': frame
        #    }, ...]
        if self.__readonly:
            for i, face in enumerate(rec_result):
                rec_result[i]['face_id'] = 0
            return

        c = self.__conn.cursor()

        c.execute('DELETE FROM files WHERE filename=?', (filename,))

        file_id = c.execute(
            'INSERT INTO files (filename) \
             VALUES (?)', (filename,)).lastrowid

        res = []
        for i, face in enumerate(rec_result):
            rec_result[i]['face_id'] = c.execute(
                'INSERT INTO faces \
                    (file_id, box, encoding, landmarks, name, dist, frame) \
                 VALUES (?, ?, ?, ?, ?, ?, ?)',
                (file_id,
                 json.dumps(face["box"]),
                 face['encoding'],
                 json.dumps(face['landmarks']),
                 face['name'],
                 face['dist'],
                 face['frame'])
            ).lastrowid

        if commit:
            self.__conn.commit()

    def remove(self, filename, commit=True):
        if self.__readonly:
            return
        c = self.__conn.cursor()
        c.execute('DELETE FROM files WHERE filename=?', (filename,))
        if commit:
            self.__conn.commit()

    def move(self, oldfilename, newfilename, commit=True):
        if self.__readonly:
            return
        c = self.__conn.cursor()
        c.execute('UPDATE files SET filename=?, synced=0 WHERE filename=?',
                  (newfilename, oldfilename))
        if commit:
            self.__conn.commit()

    def get_all_faces(self):
        c = self.__conn.cursor()
        res = c.execute('SELECT file_id, box, encoding, landmarks FROM faces')

        return [{'id': r[0],
                 'box': json.loads(r[1]),
                 'encoding': [2],
                 'landmarks': None if r[3] is None else json.loads([3])}
                for r in res.fetchall()]

    def set_name(self, face_id, name, dist, commit=True):
        if self.__readonly:
            return
        c = self.__conn.cursor()
        c.execute('UPDATE faces SET name=?, dist=? WHERE id=?',
                  (name, dist, face_id))
        if commit:
            self.__conn.commit()

    def get_names(self, filename):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT faces.name \
             FROM files JOIN faces ON files.id=faces.file_id \
             WHERE filename=?', (filename,))

        return [r[0] for r in res.fetchall()]

    def print_details(self, filename):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT faces.id, faces.box, faces.name \
             FROM files JOIN faces ON files.id=faces.file_id \
             WHERE filename=?', (filename,))

        print(f'File: {filename}')
        for r in res.fetchall():
            print(f'\tBox: {r[1]}')
            print(f'\tName: {r[2]}')

    def print_stat(self):
        c = self.__conn.cursor()

        res = c.execute(
            'SELECT name, COUNT(*) \
             FROM faces \
             GROUP BY name \
             ORDER BY 2 DESC')

        print(f'Person count')
        for r in res.fetchall():
            print(f'{r[1]}\t{r[0]}')

        res = c.execute('SELECT COUNT(*) FROM faces')
        print(f'Total faces: {res.fetchone()[0]}')

        res = c.execute('SELECT COUNT(*) FROM files')
        print(f'Total files: {res.fetchone()[0]}')

    def get_folders(self):
        c = self.__conn.cursor()
        res = c.execute('SELECT filename FROM files')

        fset = set()
        for r in res.fetchall():
            path = os.path.dirname(r[0])
            fset.add(path)
            path = os.path.dirname(path)
            fset.add(path)

        return list(fset)

    def get_files(self, folder=None):
        if folder is None:
            folder = ''
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT filename FROM files \
             WHERE filename LIKE ?', (folder + '%',))

        return [r[0] for r in res.fetchall()]

    def get_files_faces(self, where_clause, args=()):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT filename, faces.id, box, encoding, \
                    landmarks, name, dist, frame \
             FROM files JOIN faces ON files.id=faces.file_id ' +
            where_clause, args)
        return self.__build_files_faces(res.fetchall())

    def get_unmatched(self):
        return self.get_files_faces('WHERE name=""')

    def get_all(self):
        return self.get_files_faces('')

    def get_weak(self, folder):
        return self.get_files_faces(
            'WHERE filename LIKE ? AND name LIKE "%_weak"', (folder + '%',))

    def get_weak_unmatched(self, folder):
        return self.get_files_faces(
            'WHERE filename LIKE ? AND (name LIKE "%_weak" OR name = "")',
            (folder + '%',))

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
                'landmarks': None if r[4] is None else json.loads(r[4]),
                'name': r[5],
                'dist': r[6],
                'frame': r[7]})

        if filename != '':
            files_faces.append({'filename': filename, 'faces': faces})

        return files_faces

    def mark_as_synced(self, filename, commit=True):
        if self.__readonly:
            return
        c = self.__conn.cursor()
        c.execute('UPDATE files SET synced=1 WHERE filename=?', (filename,))
        if commit:
            self.__conn.commit()

    def __filenames_to_dict(self, filenames):
        res = {}
        dupl = False
        for f in filenames:
            path, name = os.path.split(f)
            if name in res:
                dupl = True
                logging.error(
                    f'Duplicate file {name} in {path} and {res[name]}')
            res[name] = path
        if dupl:
            raise Exception(f'Duplicate files')
        return res

    def update_filepaths(self, oldfolder, newfolder):
        newfiles = self.__filenames_to_dict(paths.list_images(newfolder))
        oldfiles = self.__filenames_to_dict(self.get_files(oldfolder))
        for name, oldpath in oldfiles.items():
            if name not in newfiles:
                old = os.path.join(oldpath, name)
                logging.info(f'removing unexists file: {old}')
                self.remove(old, commit=False)
            elif newfiles[name] != oldpath:
                old = os.path.join(oldpath, name)
                new = os.path.join(newfiles[name], name)
                logging.info(f'move file {old} to {new}')
                self.move(old, new, commit=False)
        self.commit()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action', help='Action', required=True,
        choices=['get_names',
                 'get_faces',
                 'print_details',
                 'print_stat',
                 'get_folders',
                 'get_files',
                 'update_filepaths'])
    parser.add_argument('-d', '--database', help='Database file')
    parser.add_argument('-f', '--file', help='File or folder')
    parser.add_argument('-l', '--logfile', help='Log file')
    parser.add_argument('--dry-run', help='Do''t modify DB',
                        action='store_true')
    return parser.parse_args()


def main():
    args = args_parse()
    log.initLogger(args.logfile)
    db = RecDB(args.database, args.dry_run)

    if args.action == 'get_names':
        print(db.get_names(args.file))
    elif args.action == 'get_faces':
        print(db.get_faces(args.file))
    elif args.action == 'print_details':
        db.print_details(args.file)
    elif args.action == 'print_stat':
        db.print_stat()
    elif args.action == 'get_folders':
        folders = db.get_folders()
        folders.sort()
        for f in folders:
            print(f)
    elif args.action == 'get_files':
        files = db.get_files(args.file)
        for f in files:
            print(f)
    elif args.action == 'update_filepaths':
        db.update_filepaths(args.file, args.file)


if __name__ == '__main__':
    main()
