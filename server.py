#!/usr/bin/python3

import os
import cgi
import json
import shutil
import urllib
import logging
import imutils
import tempfile
import argparse
import http.server
import collections

import log
import recdb
import tools
import config
import cachedb
import patterns
import recognizer


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
                    logging.debug(f'File in cache not found: {fname}')
                return
        else:
            fname = os.path.join(self.server.web_path(), path)

        self.__send_file(fname, params)

    def __send_file(self, fname, params={}):
        try:
            ext = os.path.splitext(fname)[1].lower()
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
            logging.exception(ex)

    def __send_blob(self, data, cont, params):
        if 'thumbnail' in params:
            data = tools.load_face_thumbnail(data)
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
            image_files = list(imutils.paths.list_images(cache_path))

        result = collections.defaultdict(lambda: [])
        for (i, image_file) in enumerate(image_files):
            name = image_file.split(os.path.sep)[-2]
            result[name].append(os.path.relpath(image_file, cache_path))

        res_list = [(k, sorted(r)) for k, r in result.items()]
        res_list.sort()

        self.__ok_response(res_list)

    def __get_names(self):
        self.__ok_response(self.server.names())

    def __get_name_image(self, params):
        name = params['name'][0]
        self.__send_file(self.server.name_image(name))

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
        self.__send_file(src_filename)

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
        self.server.updete_persons(params['name'][0])
        self.__ok_response('')

    def __recognize_folder_request(self, params):
        try:
            reencode = params['reencode'][0] == '1'
        except Exception:
            reencode = False
        self.server.recognize_folder(params['path'][0], reencode)
        self.__ok_response('')

    def __params_to_filter(self, params):
        fltr = {}
        for f in ('type', 'path', 'name'):
            try:
                fltr[f] = params[f][0]
            except Exception:
                fltr[f] = ''
        if fltr['type'] == '':
            raise Exception('Option type is missing')
        return fltr

    def __generate_faces_request(self, params):
        fltr = self.__params_to_filter(params)
        self.server.save_faces(fltr)
        self.__ok_response('')

    def __match_request(self, params):
        try:
            save_faces = params['save_faces'][0] == '1'
        except Exception:
            save_faces = False
        fltr = self.__params_to_filter(params)
        self.server.match(fltr, save_faces)
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

    def do_GET(self):
        logging.debug('do_GET: ' + self.path)
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

            if path == '/':
                path = 'index.html'

            if '..' in path:
                logging.warning('".." in path: ' + path)
                self.__not_found_response()
                return

            ext = os.path.splitext(path)[1].lower()
            if ext in ('.html', '.js', '.css', '.png', '.jpg'):
                self.__file_request(path, params)
                return

            logging.warning('Wrong path: ' + path)
            self.__not_found_response()
        except Exception as ex:
            self.__server_error_response(str(ex))
            logging.exception(ex)

    def do_POST(self):
        logging.debug('do_POST: ' + self.path)
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

        except Exception as ex:
            self.__server_error_response(str(ex))
            logging.exception(ex)


class FaceRecServer(http.server.HTTPServer):
    def __init__(self, cfg):
        self.__status = {'state': ''}
        self.__recognizer = None
        self.__cfg = cfg
        self.__patterns = patterns.Patterns(
            cfg['main']['patterns'],
            model=cfg['main']['model'],
            max_size=cfg['main']['max_image_size'],
            num_jitters=cfg['main']['num_jitters'],
            encoding_model=cfg['main']['encoding_model'])

        self.__patterns.load()
        self.__load_patterns_persons()

        self.__db = recdb.RecDB(cfg['main']['db'])

        cachedb_file = cfg['main']['cachedb']
        if cachedb_file:
            logging.info(f'Using cachedb: {cachedb_file}')
            self.__cdb = cachedb.CacheDB(cachedb_file)
        else:
            logging.info(f'Not using cachedb')
            self.__cdb = None

        port = int(cfg['server']['port'])
        self.__web_path = cfg['server']['web_path']
        self.__face_cache_path = cfg['server']['face_cache_path']
        super().__init__(('', port), FaceRecHandler)

    def __start_recognizer(self, method, *args):
        self.status()
        if self.__recognizer is not None:
            logging.warning('Trying to create second recognizer')
            raise Exception('Recognizer already started')

        self.__clean_cache()
        tools.cuda_init()
        self.__recognizer = recognizer.createRecognizer(
            self.__patterns, self.__cfg, self.__cdb)
        self.__recognizer.start_method(method, *args)

    def __generate_patterns(self):
        self.__status = {'state': 'patterns_generation'}
        self.__patterns.generate()
        self.__load_patterns_persons()

    def updete_persons(self, name):
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
                tools.cuda_release()

        return self.__status

    def __clean_cache(self):
        if self.__cdb is not None:
            self.__cdb.clean_cache()
        if os.path.exists(self.__face_cache_path):
            shutil.rmtree(self.__face_cache_path)

    def recognize_folder(self, path, reencode):
        self.__generate_patterns()
        self.__start_recognizer('recognize_folder',
                                path, self.__db, self.__face_cache_path,
                                reencode)

    def match(self, fltr, save_faces):
        self.__generate_patterns()
        self.__start_recognizer('match',
                                self.__db, fltr, self.__face_cache_path,
                                save_faces)

    def clusterize(self, fltr):
        self.__generate_patterns()
        self.__start_recognizer('clusterize',
                                self.__db, fltr, self.__face_cache_path)

    def save_faces(self, fltr):
        self.__start_recognizer('save_faces',
                                self.__db, fltr, self.__face_cache_path)

    def get_faces_by_face(self, filename):
        self.__start_recognizer('get_faces_by_face',
                                self.__db, filename, self.__face_cache_path,
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
        logfile = cfg['server']['log_file']

    log.initLogger(logfile)

    try:
        server = FaceRecServer(cfg)
        logging.info("Face rec server up.")
        server.serve_forever()
    except KeyboardInterrupt:
        server.socket.close()
        logging.info("Face rec server down.")


if __name__ == '__main__':
    main()
