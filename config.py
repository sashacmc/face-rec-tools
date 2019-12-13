#!/usr/bin/python3

import os
import configparser


class Config(object):
    DEFAULT_CONFIG_FILE = os.path.expanduser('~/.face-rec.cfg')
    DEFAULTS = {
        'main': {
            'model': 'cnn',
            'num_jitters': 1,
            'threshold': 0.5,
            'max_size': 1000,
            'db': '/mnt/multimedia/recdb/rec.db',
            'patterns': '/mnt/multimedia/recdb/face_rec_patt/',
        },
        'server': {
            'port': 8081,
            'web_path': 'web',
            'face_cache_path': '/tmp/facereccache/',
            'log_file': 'face-rec-server.log',
        }
    }

    def __init__(self, filename=None, create=False):
        if filename is None:
            filename = self.DEFAULT_CONFIG_FILE

        self.__config = configparser.ConfigParser()
        self.__config.read_dict(self.DEFAULTS)
        self.__config.read([filename, ])

        if create:
            self.__create_if_not_exists()

    def __create_if_not_exists(self):
        if os.path.exists(self.DEFAULT_CONFIG_FILE):
            return

        with open(self.DEFAULT_CONFIG_FILE, 'w') as conffile:
            self.__config.write(conffile)

    def __getitem__(self, sect):
        return self.__config[sect]


if __name__ == "__main__":
    Config(create=True)
