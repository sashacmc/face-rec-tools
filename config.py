#!/usr/bin/python3

import os
import configparser


class Config(object):
    DEFAULT_CONFIG_FILE = os.path.expanduser('~/.face-rec.cfg')
    DEFAULTS = {
        'main': {
            'model': 'cnn',
            'num_jitters': 100,
            'threshold': 0.3,
            'threshold_weak': 0.35,
            'threshold_clusterize': 0.4,
            'max_image_size': 1000,
            'min_face_size': 20,
            'debug_out_image_size': 100,
            'encoding_model': 'large',
            'max_workers': 2,
            'db': '/mnt/multimedia/recdb/rec.db',
            'cachedb': '/opt/tmp/facereccache.db',
            'patterns': '/mnt/multimedia/recdb/face_rec_patt/',
        },
        'server': {
            'port': 8081,
            'web_path': 'web',
            'face_cache_path': '/tmp/facereccache/',
            'log_file': 'face-rec-server.log',
        },
        'plex': {
            'db': '/opt/plexmediaserver/Library/Application Support/Plex Media Server/Plug-in Support/Databases/com.plexapp.plugins.library.db',
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

    def get_def(self, sect, name, default):
        if default is not None:
            return default

        return self.__config[sect][name]


if __name__ == "__main__":
    Config(create=True)
