#!/usr/bin/python3

import time
import logging
import sqlite3


class PlexDB(object):
    def __init__(self, filename):
        self.__conn = sqlite3.connect(filename)

    def set_tags(self, filename, tags):
        fid = self.__get_id(filename)
        if fid is None:
            logging.warning(f'Filename not found: {filename}')
            return False

        for tag in tags:
            tag_id = self.__get_tag_id(tag)
            if tag_id is None:
                logging.warning(f'Tag not found: {tag}')
                continue

            self.__set_tag(fid, tag_id)

        return True

    def get_tags(self, filename):
        fid = self.__get_id(filename)
        if fid is None:
            logging.warning(f'Filename not found: {filename}')
            return None

        c = self.__conn.cursor()
        res = c.execute(
            'SELECT tags.tag \
             FROM taggings \
             JOIN tags ON tags.id=taggings.tag_id \
             WHERE taggings.metadata_item_id=?', (fid,))

        return [r[0] for r in res.fetchall()]

    def clean_tags(self, filename, tags):
        fid = self.__get_id(filename)
        if fid is None:
            logging.warning(f'Filename not found: {filename}')
            return 0

        c = self.__conn.cursor()

        res = 0
        for tag in tags:
            tag_id = self.__get_tag_id(tag)
            res += self.__clean_tag(fid, tag_id)

        self.__conn.commit()

        logging.debug(f'Removed {res} tags for {filename}')

        return res

    def create_tag(self, tag):
        c = self.__conn.cursor()

        tm = self.__gen_time()

        res = c.execute(
            'INSERT INTO tags \
             (tag, tag_type, created_at, updated_at) \
             VALUES (?,0,?,?)',
            (tag, tm, tm)).lastrowid

        self.__conn.commit()

        return res

    def create_tag(self, tag):
        c = self.__conn.cursor()

        tm = self.__gen_time()

        res = c.execute(
            'INSERT INTO tags \
             (tag, tag_type, created_at, updated_at) \
             VALUES (?,0,?,?)',
            (tag, tm, tm)).lastrowid

        self.__conn.commit()

        return res

    def delete_tag(self, tag_id):
        c = self.__conn.cursor()

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

        self.__conn.commit()

        return res

    def __set_tag(self, fid, tag_id):
        c = self.__conn.cursor()

        tm = self.__gen_time()

        c.execute(
            'INSERT INTO taggings \
             (metadata_item_id, tag_id, "index", created_at) \
             VALUES (?,?,0,?)',
            (fid, tag_id, tm))

        self.__conn.commit()

    def __clean_tag(self, fid, tag_id):
        c = self.__conn.cursor()

        res = c.execute(
            'DELETE FROM taggings \
             WHERE metadata_item_id=? AND tag_id=?',
            (fid, tag_id)).rowcount

        self.__conn.commit()

        return res

    def __get_tag_id(self, tag):
        c = self.__conn.cursor()
        res = c.execute('SELECT id FROM tags WHERE tag=?', (tag,))

        row = res.fetchone()
        if row:
            return row[0]
        else:
            return None

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
    logging.basicConfig(level=logging.DEBUG)
    db = PlexDB('/opt/tmp/test.db')
    fn = '/mnt/multimedia/NEW/Foto/2019/2019-11-05/2019-11-05_18-19-12.JPG'
    print(db.get_tags(fn))