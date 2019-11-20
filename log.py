import os
import sys
import logging

LOGFMT = '[%(asctime)s] [%(levelname)s] %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'


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
    logging.getLogger().addHandler(fh)

    logging.getLogger().setLevel(logging.INFO)

    logging.info('Log file: ' + str(filename))
    logging.debug(str(sys.argv))
