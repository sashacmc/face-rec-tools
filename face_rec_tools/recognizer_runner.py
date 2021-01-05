#!/usr/bin/python3

import os
import sys
import time
import multiprocessing

sys.path.insert(0, os.path.abspath('..'))

from face_rec_tools import log  # noqa
from face_rec_tools import tools  # noqa


class RecognizerRunner(multiprocessing.Process):
    def __init__(self, config, method, *args):
        multiprocessing.Process.__init__(self)
        self.__config = config
        self.__method = method
        self.__args = args
        self.__manager = multiprocessing.Manager()
        self.__status = self.__manager.dict(tools.RECOGNIZER_STATUS_INIT)

    def run(self):
        os.nice(10)
        try:
            from face_rec_tools import recdb  # noqa
            from face_rec_tools import config  # noqa
            from face_rec_tools import cachedb  # noqa
            from face_rec_tools import patterns  # noqa
            from face_rec_tools import recognizer  # noqa

            cfg = config.Config(self.__config)

            patterns = patterns.createPatterns(cfg)

            db = recdb.RecDB(cfg.get_path('files', 'db'))
            cdb = cachedb.createCacheDB(cfg)
            cuda_memory_limit = int(cfg['processing']['cuda_memory_limit'])

            patterns.load()
            recognizer = recognizer.createRecognizer(
                patterns, cfg, cdb, db, self.__status)

            log.info(f'Run in process: {self.__method}{self.__args}')

            if self.__method == 'recognize_folder':
                tools.cuda_init(cuda_memory_limit)
                recognizer.recognize_folder(*self.__args)
            elif self.__method == 'match':
                tools.cuda_init(cuda_memory_limit)
                recognizer.match(*self.__args)
            elif self.__method == 'clusterize':
                recognizer.clusterize(*self.__args)
            elif self.__method == 'save_faces':
                recognizer.save_faces(*self.__args)
            elif self.__method == 'get_faces_by_face':
                tools.cuda_init(cuda_memory_limit)
                recognizer.get_faces_by_face(*self.__args)
            log.info(f'Process done: {self.__method}')
            self.__status['state'] = 'done'
        except Exception as ex:
            log.exception(ex)
            self.__status['state'] = 'error'
            self.__status['error'] = str(ex)

    def status(self):
        status = dict(self.__status)
        if status['current'] > 0:
            elap_time = time.time() - status['starttime']
            est_time = \
                (status['count'] - status['current']) \
                / status['current'] * elap_time
            status['estimation'] = tools.seconds_to_str(est_time)
            status['elapsed'] = tools.seconds_to_str(elap_time)
        else:
            status['estimation'] = ''
            status['elapsed'] = ''
        return status

    def stop(self, save):
        log.info(f'Runner stop called ({save})')
        self.__status['stop'] = True
        self.__status['save'] = save
