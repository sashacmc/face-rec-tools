#!/usr/bin/python3

import unittest

import plexdb


class TestPlexDB(unittest.TestCase):
    def setUp(self):
        self.db = plexdb.PlexDB('/opt/tmp/test.db')
        self.file = \
            '/mnt/multimedia/NEW/Foto/2019/2019-11-05/2019-11-05_18-19-12.JPG'

    def test_create_delete_tag(self):
        self.db.create_tag('TestPlexDB_test_tag')
        self.db.delete_tag('TestPlexDB_test_tag')

    def test_workflow(self):
        new_tags = ('test',)
        self.db.clean_tags(self.file, new_tags)
        initial_tags = self.db.get_tags(self.file)
        print(initial_tags)
        self.assertNotEqual(len(initial_tags), 0)

        self.db.set_tags(self.file, new_tags)
        tags = self.db.get_tags(self.file)
        print(tags)
        self.assertEqual(set(tags) - set(initial_tags), set(new_tags))
        self.assertEqual(len(tags), len(initial_tags) + len(new_tags))

        self.db.clean_tags(self.file, new_tags)
        tags = self.db.get_tags(self.file)
        print(tags)
        self.assertEqual(initial_tags, tags)


if __name__ == '__main__':
    unittest.main()