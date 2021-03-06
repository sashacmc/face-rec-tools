#!/usr/bin/python3

import os
import re
import cgi
import sys
import json
import shutil
import urllib
import tempfile
import argparse
import http.server
import collections

sys.path.insert(0, os.path.abspath('..'))

from face_rec_tools import log  # noqa
from face_rec_tools import recdb  # noqa
from face_rec_tools import tools  # noqa
from face_rec_tools import config  # noqa
from face_rec_tools import cachedb  # noqa
from face_rec_tools import patterns  # noqa
from face_rec_tools import recognizer_runner  # noqa


class FaceRecHandler(http.server.BaseHTTPRequestHandler):

    def __ok_response(self, result):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(bytearray(json.dumps(result), 'utf-8'))

    def __text_response(self, result):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(bytearray(result, 'utf-8'))

    def __bad_request_response(self, err):
        self.send_response(400)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(bytearray(err, 'utf-8'))

    def __not_found_response(self):
        self.send_error(404, 'File Not Found: %s' % self.path)

    def __server_error_response(self, err):
        self.send_error(500, 'Internal Server Error: %s' % err)

    def __file_request(self, path, params):
        if path[0] == '/':
            path = path[1:]

        path = urllib.parse.unquote(path)

        if path.startswith('cache/'):
            fname = os.path.join(self.server.face_cache_path(), path[6:])
            if self.server.cdb() is not None:
                data = self.server.cdb().get_from_cache(fname)
                if data is not None:
                    self.__send_blob(data, 'image/jpeg', params)
                else:
                    self.__not_found_response()
                    log.debug(f'File in cache not found: {fname}')
                return
        elif path.startswith('pattern/'):
            fname = self.server.patterns().fullpath(path[8:])
        else:
            fname = os.path.join(self.server.web_path(), path)

        self.__send_file(fname, params)

    def __send_file(self, fname, params={}):
        try:
            ext = tools.get_low_ext(fname)
            cont = ''
            if ext == '.html':
                cont = 'text/html'
            elif ext == '.js':
                cont = 'text/javascript'
            elif ext == '.css':
                cont = 'text/css'
            elif ext == '.png':
                cont = 'image/png'
            elif ext == '.jpg':
                cont = 'image/jpeg'
            else:
                cont = 'text/none'
            with open(fname, 'rb') as f:
                self.__send_blob(f.read(), cont, params)
        except IOError as ex:
            self.__not_found_response()
            log.exception(ex)

    def __send_blob(self, data, cont, params):
        if 'thumbnail' in params:
            if params['thumbnail'][0] == 'on':
                data = tools.load_face_thumbnail(data)
            elif params['thumbnail'][0] == 'prefer':
                th_data = tools.load_face_thumbnail(data)
                if th_data:
                    data = th_data
        self.send_response(200)
        self.send_header('Content-type', cont)
        self.end_headers()
        self.wfile.write(bytearray(data))

    def __data(self):
        datalen = int(self.headers['Content-Length'])
        data_raw = self.rfile.read(datalen)
        data = data_raw.decode('utf-8')
        return urllib.parse.parse_qs(data)

    def __form_data(self):
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        form = cgi.parse_multipart(self.rfile, pdict)
        return form

    def __path_params(self):
        path_params = self.path.split('?')
        if len(path_params) > 1:
            return path_params[0], urllib.parse.parse_qs(path_params[1])
        else:
            return path_params[0], {}

    def __list_cache(self, params):
        cache_path = self.server.face_cache_path()
        if self.server.cdb() is not None:
            image_files = self.server.cdb().list_cache()
        else:
            image_files = tools.list_files(cache_path, tools.IMAGE_EXTS)

        result = collections.defaultdict(lambda: [])
        for (i, image_file) in enumerate(image_files):
            name = image_file.split(os.path.sep)[-2]
            result[name].append(os.path.relpath(image_file, cache_path))

        if 'sort' in params and params['sort'][0] == 'date':
            def key(s): return re.sub('_\d\d\d_', '_', s)
        else:
            key = None
        res_list = [(k, sorted(r, key=key)) for k, r in result.items()]
        res_list.sort()

        self.__ok_response(res_list)

    def __clean_cache(self):
        self.server.clean_cache()
        self.__ok_response('')

    def __get_names(self):
        self.__ok_response(self.server.names())

    def __get_name_image(self, params):
        name = params['name'][0]
        self.__send_file(self.server.name_image(name),
                         {'thumbnail': ['prefer']})

    def __get_face_file_description(self, path):
        if path.startswith('cache/'):
            fname = os.path.join(self.server.face_cache_path(), path[6:])
            if self.server.cdb() is not None:
                data = self.server.cdb().get_from_cache(fname)
                if data is not None:
                    return tools.load_face_description(data)[0]
            else:
                return tools.load_face_description(fname)[0]
        return None

    def __get_face_src(self, params):
        path = params['path'][0]
        descr = self.__get_face_file_description(path)
        if descr is None:
            self.__not_found_response()
        src_filename = descr.get('src', None)
        if src_filename is None:
            self.__not_found_response()
        if 'type' in params and params['type'][0] == 'info':
            ext = tools.get_low_ext(src_filename)
            tp = ''
            if ext in tools.IMAGE_EXTS:
                tp = 'image'
            elif ext in tools.VIDEO_EXTS:
                tp = 'video'
            self.__ok_response({
                'filename': src_filename,
                'type': tp,
                'names': self.server.db().get_names(src_filename)
            })
        else:
            self.__send_file(src_filename)

    def __get_face_pattern(self, params):
        path = params['path'][0]
        descr = self.__get_face_file_description(path)
        if descr is None:
            self.__not_found_response()
            return
        face_id = descr.get('face_id', None)
        if face_id is None:
            log.warning(f'face file {path} without face_id')
            self.__not_found_response()
            return
        count, ff = self.server.db().get_face(face_id)
        if count == 0:
            log.warning(f'face with id {face_id} not found')
            self.__not_found_response()
            return
        pattern_filename = next(ff)['faces'][0]['pattern']
        if pattern_filename == '':
            log.warning(f'pattern file not specified')
            self.__not_found_response()
            return
        self.__ok_response(pattern_filename)

    def __get_folders(self):
        self.__ok_response(sorted(self.server.db().get_folders()))

    def __get_status(self):
        self.__ok_response(self.server.status())

    def __add_to_pattern_request(self, params, data):
        cache_path = self.server.face_cache_path()
        files = data['files'][0].split('|')
        filenames = [os.path.join(cache_path, f) for f in files]
        if self.server.cdb() is not None:
            for fn in filenames:
                data = self.server.cdb().get_from_cache(fn)
                if data is None:
                    raise Exception(f'No data for file {fn}')
                self.server.patterns().add_file_data(params['name'][0],
                                                     fn, data,
                                                     params['bad'][0] == '1')
                self.server.cdb().remove_from_cache(fn)
            self.server.cdb().commit()
        else:
            self.server.patterns().add_files(params['name'][0],
                                             filenames, True, True,
                                             params['bad'][0] == '1')
        self.server.update_persons(params['name'][0])
        self.__ok_response('')

    def __recognize_folder_request(self, params):
        reencode = params.get('reencode', ('',))[0] == '1'
        skip_face_gen = params.get('skip_face_gen', ('',))[0] == '1'

        self.server.recognize_folder(
            os.path.expanduser(params['path'][0]), reencode, skip_face_gen)
        self.__ok_response('')

    def __params_to_filter(self, params):
        fltr = {}
        for f in ('type', 'path', 'name'):
            fltr[f] = params.get(f, ('',))[0]

        if fltr['type'] == '':
            raise Exception('Option type is missing')
        return fltr

    def __generate_faces_request(self, params):
        fltr = self.__params_to_filter(params)
        self.server.save_faces(fltr)
        self.__ok_response('')

    def __match_request(self, params):
        save_faces = params.get('save_faces', ('',))[0] == '1'
        skip_face_gen = params.get('skip_face_gen', ('',))[0] == '1'

        fltr = self.__params_to_filter(params)
        self.server.match(fltr, save_faces, skip_face_gen)
        self.__ok_response('')

    def __clusterize_request(self, params):
        fltr = self.__params_to_filter(params)
        self.server.clusterize(fltr)
        self.__ok_response('')

    def __get_faces_by_face_request(self, params, data):
        tf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tf.write(data['file'][0])
        tf.close()
        self.server.get_faces_by_face(tf.name)
        self.__ok_response('')

    def __stop_request(self, params):
        save = params.get('save', ('',))[0] == '1'
        self.server.stop(save)
        self.__ok_response('')

    def do_GET(self):
        log.debug('do_GET: ' + self.path)
        try:
            path, params = self.__path_params()
            if path == '/list_cache':
                self.__list_cache(params)
                return

            if path == '/get_names':
                self.__get_names()
                return

            if path == '/get_name_image':
                self.__get_name_image(params)
                return

            if path == '/get_folders':
                self.__get_folders()
                return

            if path == '/get_status':
                self.__get_status()
                return

            if path == '/get_face_src':
                self.__get_face_src(params)
                return

            if path == '/get_face_pattern':
                self.__get_face_pattern(params)
                return

            if path == '/':
                path = 'index.html'

            if '..' in path:
                log.warning('".." in path: ' + path)
                self.__not_found_response()
                return

            ext = tools.get_low_ext(path)
            if ext in ('.html', '.js', '.css', '.png', '.jpg'):
                self.__file_request(path, params)
                return

            log.warning('Wrong path: ' + path)
            self.__not_found_response()
        except Exception as ex:
            self.__server_error_response(str(ex))
            log.exception(ex)

    def do_POST(self):
        log.debug('do_POST: ' + self.path)
        try:
            path, params = self.__path_params()

            if path == '/add_to_pattern':
                self.__add_to_pattern_request(params, self.__data())
                return

            if path == '/recognize_folder':
                self.__recognize_folder_request(params)
                return

            if path == '/generate_faces':
                self.__generate_faces_request(params)
                return

            if path == '/match':
                self.__match_request(params)
                return

            if path == '/clusterize':
                self.__clusterize_request(params)
                return

            if path == '/get_faces_by_face':
                self.__get_faces_by_face_request(params, self.__form_data())
                return

            if path == '/stop':
                self.__stop_request(params)
                return

            if path == '/clean_cache':
                self.__clean_cache()
                return

        except Exception as ex:
            self.__server_error_response(str(ex))
            log.exception(ex)


class FaceRecServer(http.server.HTTPServer):
    def __init__(self, cfg):
        self.__status = {'state': ''}
        self.__recognizer = None
        self.__cfg = cfg

        self.__patterns = patterns.createPatterns(cfg)
        self.__patterns.load()
        self.__load_patterns_persons()

        self.__db = recdb.RecDB(cfg.get_path('files', 'db'))
        self.__cdb = cachedb.createCacheDB(cfg)

        port = int(cfg['server']['port'])

        self.__web_path = cfg.get_data_path('server', 'web_path')

        self.__face_cache_path = cfg['server']['face_cache_path']
        super().__init__(('', port), FaceRecHandler)

    def __start_recognizer(self, method, *args):
        self.status()
        if self.__recognizer is not None:
            log.warning('Trying to create second recognizer')
            raise Exception('Recognizer already started')

        self.__recognizer = recognizer_runner.RecognizerRunner(
            self.__cfg.filename(), method, *args)
        self.__recognizer.start()

    def __generate_patterns(self):
        self.__status = {'state': 'patterns_generation'}
        self.__patterns.generate()
        self.__load_patterns_persons()

    def update_persons(self, name):
        if name not in self.__names:
            self.__patterns.generate()
            self.__load_patterns_persons()

    def __load_patterns_persons(self):
        self.__names = [
            p['name'] for p in self.__patterns.persons()
        ]
        self.__name_images = {
            p['name']: self.__patterns.fullpath(p['image'])
            for p in self.__patterns.persons()
        }

    def web_path(self):
        return self.__web_path

    def face_cache_path(self):
        return self.__face_cache_path

    def patterns(self):
        return self.__patterns

    def names(self):
        return self.__names

    def name_image(self, name):
        return self.__name_images[name]

    def db(self):
        return self.__db

    def cdb(self):
        return self.__cdb

    def status(self):
        if self.__recognizer:
            self.__status = self.__recognizer.status()
            if self.__status['state'] in ('done', 'error'):
                self.__recognizer.join()
                self.__recognizer = None

        return self.__status

    def stop(self, save):
        if self.__recognizer:
            self.__recognizer.stop(save)

    def clean_cache(self):
        if self.__cdb is not None:
            self.__cdb.clean_cache()
        elif os.path.exists(self.__face_cache_path):
            shutil.rmtree(self.__face_cache_path)

    def recognize_folder(self, path, reencode, skip_face_gen):
        self.__generate_patterns()
        if not skip_face_gen:
            self.clean_cache()
        self.__start_recognizer('recognize_folder',
                                path, self.__face_cache_path,
                                reencode,
                                skip_face_gen)

    def match(self, fltr, save_faces, skip_face_gen):
        self.__generate_patterns()
        if not skip_face_gen:
            self.clean_cache()
        self.__start_recognizer('match',
                                fltr, self.__face_cache_path,
                                save_faces,
                                skip_face_gen)

    def clusterize(self, fltr):
        self.clean_cache()
        self.__start_recognizer('clusterize',
                                fltr, self.__face_cache_path)

    def save_faces(self, fltr):
        self.clean_cache()
        self.__start_recognizer('save_faces',
                                fltr, self.__face_cache_path)

    def get_faces_by_face(self, filename):
        self.clean_cache()
        self.__start_recognizer('get_faces_by_face',
                                filename, self.__face_cache_path,
                                True)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('-l', '--logfile', help='Log file')
    return parser.parse_args()


def main():
    args = args_parse()

    cfg = config.Config(args.config)

    if args.logfile:
        logfile = args.logfile
    else:
        logfile = cfg.get_path('server', 'log_file')

    log.initLogger(logfile)

    try:
        server = FaceRecServer(cfg)
        log.info("Face rec server up.")
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop(False)
        server.status()
        server.server_close()
        log.info("Face rec server down.")


if __name__ == '__main__':
    main()
