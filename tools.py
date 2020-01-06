import cv2
import piexif
import pickle
import logging
from PIL import Image


def read_image(image_file, max_size):
    image = cv2.imread(image_file)

    height, width, col = image.shape

    if height > width:
        scale = max_size / height
    else:
        scale = max_size / width

    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_face(out_filename, image, box, encoding, out_size):
    top, right, bottom, left = box
    d = (bottom - top) // 2
    out_image = image[
        max(0, top - d):bottom + d,
        max(0, left - d):right + d]

    encd = pickle.dumps({'encoding': encoding}, protocol=0)
    exif = piexif.dump(
        {"0th": {piexif.ImageIFD.ImageDescription: encd}})
    im = Image.fromarray(out_image)
    im.thumbnail((out_size, out_size))
    im.save(out_filename, exif=exif, format="JPEG")


class LazyImage(object):
    def __init__(self, image_file, max_size):
        self.__image_file = image_file
        self.__max_size = max_size
        self.__image = None

    def get(self):
        if self.__image is None:
            logging.debug(f'LazyImage load: {self.__image_file}')
            self.__image = read_image(self.__image_file, self.__max_size)
        return self.__image
