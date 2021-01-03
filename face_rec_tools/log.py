import os
import sys
import logging

LOGFMT = '[%(asctime)s] [%(levelname)s] %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
MODULE = 'face-rec-tools'


def initLogger(filename=None):
    if filename is not None:
        try:
            os.makedirs(os.path.split(filename)[0])
        except OSError:
            pass
        fh = logging.FileHandler(filename, 'a')
    else:
        fh = logging.StreamHandler()

    fmt = logging.Formatter(LOGFMT, DATEFMT)
    fh.setFormatter(fmt)
    logging.getLogger(MODULE).addHandler(fh)

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger(MODULE).setLevel(logging.DEBUG)

    info('Log file: ' + str(filename))
    debug(str(sys.argv))


def debug(msg, *args, **kwargs):
    logging.getLogger(MODULE).debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    logging.getLogger(MODULE).info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    logging.getLogger(MODULE).warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    logging.getLogger(MODULE).error(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    logging.getLogger(MODULE).exception(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    logging.getLogger(MODULE).critical(msg, *args, **kwargs)
