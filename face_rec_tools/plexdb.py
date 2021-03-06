#!/usr/bin/python3

import os
import sys
import time
import sqlite3

sys.path.insert(0, os.path.abspath('..'))

from face_rec_tools import log  # noqa

TAG_TYPE_PHOTO = 0
TAG_TYPE_VIDEO = 2


class PlexDB(object):
    def __init__(self, filename, readonly=False):
        log.debug(f'Connect to {filename} ({readonly})')
        self.__conn = sqlite3.connect(
            'file:' + filename + ('?mode=ro' if readonly else ''),
            uri=True)
        self.__tag_cache = {}
        self.__readonly = readonly

    def commit(self):
        if self.__readonly:
            return
        self.__conn.commit()

    def get_files(self, folder):
        c = self.__conn.cursor()
        res = c.execute('SELECT file FROM media_parts \
                         WHERE file LIKE ? \
                         ORDER BY file',
                        (folder + '%',))
        return [row[0] for row in res.fetchall()]

    def set_tags(self, filename, tags, tag_type, commit=True):
        fid = self.__get_id(filename)
        if fid is None:
            log.warning(f'Filename not found: {filename}')
            return False

        for tag in tags:
            tag_id = self.__get_tag_id(tag, tag_type)
            if tag_id is None:
                log.warning(f'Tag not found: {tag}')
                continue

            self.__set_tag(fid, tag_id, commit)

        return True

    def get_tags(self, filename):
        fid = self.__get_id(filename)
        if fid is None:
            log.warning(f'Filename not found: {filename}')
            return None

        c = self.__conn.cursor()
        res = c.execute(
            'SELECT tags.tag \
             FROM taggings \
             JOIN tags ON tags.id=taggings.tag_id \
             WHERE taggings.metadata_item_id=?', (fid,))

        return [r[0] for r in res.fetchall()]

    def clean_tags(self, filename, tags=None, tag_prefix=None, commit=True):
        fid = self.__get_id(filename)
        if fid is None:
            log.warning(f'Filename not found: {filename}')
            return 0

        res = 0
        if tags is not None:
            for tag in tags:
                res += self.__clean_tag(fid,
                                        self.__get_tag_id(tag,
                                                          TAG_TYPE_PHOTO),
                                        commit)
                res += self.__clean_tag(fid,
                                        self.__get_tag_id(tag,
                                                          TAG_TYPE_VIDEO),
                                        commit)
        if tag_prefix is not None:
            tag_ids = self.__get_tag_ids(tag_prefix)
            for tag_id in tag_ids:
                res += self.__clean_tag(fid, tag_id, commit)

        if commit:
            self.__conn.commit()

        log.debug(f'Removed {res} tags for {filename}')

        return res

    def create_tag(self, tag, tag_type, commit=True):
        if self.__readonly:
            return

        c = self.__conn.cursor()

        tm = self.__gen_time()

        res = c.execute(
            'INSERT INTO tags \
             (tag, tag_type, created_at, updated_at) \
             VALUES (?,?,?,?)',
            (tag, tag_type, tm, tm)).lastrowid

        if commit:
            self.__conn.commit()

        return res

    def delete_tag(self, tag, tag_type, commit=True):
        tag_id = self.__get_tag_id(tag, tag_type)
        if tag_id is None:
            return 0

        return self.__delete_tag(tag_id, commit)

    def delete_tags(self, tag_prefix, cleanup=False, commit=True):
        tag_ids = self.__get_tag_ids(tag_prefix)
        res = 0
        for tag_id in tag_ids:
            res += self.__delete_tag(tag_id, cleanup, commit)

        return res

    def __delete_tag(self, tag_id, cleanup=False, commit=True):
        if self.__readonly:
            return
        c = self.__conn.cursor()

        if cleanup:
            res = c.execute(
                'DELETE FROM taggings \
                 WHERE tag_id=?',
                (tag_id, )).rowcount
            if res != 0:
                log.debug(f'Removed {res} taggings for tag {tag_id}')
        else:
            res = c.execute(
                'SELECT count(*) FROM taggings \
                 WHERE tag_id=?',
                (tag_id, ))

            count = res.fetchone()[0]
            if count != 0:
                raise Exception(f'Found {count} taggings for tag {tag_id}')

        res = c.execute(
            'DELETE FROM tags \
             WHERE id=?',
            (tag_id, )).rowcount

        if commit:
            self.__conn.commit()

        return res

    def tag_exists(self, tag, tag_type):
        return not self.__get_tag_id(tag, tag_type) is None

    def __set_tag(self, fid, tag_id, commit=True):
        if self.__readonly:
            return
        c = self.__conn.cursor()

        tm = self.__gen_time()

        c.execute(
            'INSERT INTO taggings \
             (metadata_item_id, tag_id, "index", created_at) \
             VALUES (?,?,0,?)',
            (fid, tag_id, tm))

        if commit:
            self.__conn.commit()

    def __clean_tag(self, fid, tag_id, commit=True):
        if self.__readonly:
            return
        c = self.__conn.cursor()

        res = c.execute(
            'DELETE FROM taggings \
             WHERE metadata_item_id=? AND tag_id=?',
            (fid, tag_id)).rowcount

        if commit:
            self.__conn.commit()

        return res

    def __get_tag_id(self, tag, tag_type):
        if tag in self.__tag_cache:
            return self.__tag_cache[tag]

        c = self.__conn.cursor()
        res = c.execute('SELECT id FROM tags WHERE tag_type=? AND tag=?',
                        (tag_type, tag))

        row = res.fetchone()
        if row:
            tag_id = row[0]
        else:
            tag_id = None

        return tag_id

    def __get_tag_ids(self, tag_prefix):
        c = self.__conn.cursor()
        res = c.execute('SELECT id FROM tags WHERE tag LIKE ?',
                        (tag_prefix + '%',))

        return [row[0] for row in res.fetchall()]

    def __get_id(self, filename):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT metadata_item_id \
             FROM media_parts \
             JOIN media_items ON media_item_id=media_items.id \
             WHERE file=?', (filename,))

        row = res.fetchone()
        if row:
            return row[0]
        else:
            return None

    def __get_filename(self, fid):
        c = self.__conn.cursor()
        res = c.execute(
            'SELECT file \
             FROM media_parts \
             JOIN media_items ON media_item_id=media_items.id \
             WHERE metadata_item_id=?', (fid,))

        row = res.fetchone()
        if row:
            return row[0]
        else:
            return None

    def __gen_time(self):
        return time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(time.time()))


if __name__ == '__main__':
    db = PlexDB('/opt/tmp/test.db')
    fn = '/mnt/multimedia/NEW/Foto/2019/2019-11-05/2019-11-05_18-19-12.JPG'
    print(db.get_tags(fn))
