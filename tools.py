import io
import cv2
import piexif
import pickle
import logging
from PIL import Image, ImageDraw


def read_image(image_file, max_size):
    image = cv2.imread(image_file)
    return prepare_image(image, max_size)


def prepare_image(image, max_size):
    height, width, col = image.shape

    if height > width:
        scale = max_size / height
    else:
        scale = max_size / width

    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def load_face_description(filename):
    try:
        exif = piexif.load(filename)
        encd = exif["0th"][piexif.ImageIFD.ImageDescription]
        descr = pickle.loads(encd)
        thumbnail = exif.pop('thumbnail')
        if thumbnail is not None:
            thumbnail = Image.open(io.BytesIO(thumbnail))
        return descr, thumbnail
    except Exception as ex:
        return None, None


def save_with_description(image, descr, thumbnail, filename):
    thumbnail_data = None
    if thumbnail is not None:
        o = io.BytesIO()
        thumbnail.save(o, format="JPEG", quality=90)
        thumbnail_data = o.getvalue()
    encd = pickle.dumps(descr, protocol=0)
    exif = piexif.dump({"0th": {piexif.ImageIFD.ImageDescription: encd},
                        "thumbnail": thumbnail_data, "1st": {}})
    image.save(filename, exif=exif, format="JPEG", quality=90)


def __set_landmarks(image, face_landmarks):
    for pts in face_landmarks.values():
        for pt in pts:
            try:
                r, g, b = image.getpixel(pt)
                image.putpixel(pt, ((128 + r) % 255,
                                    (128 + g) % 255,
                                    (128 + b) % 255))
            except IndexError:
                logging.debug(f'Incorrect landmark point: {pt}')


def __set_landmarks_lines(image, face_landmarks):
    draw = ImageDraw.Draw(image)
    for pts in face_landmarks.values():
        draw.line(pts, fill=(255, 255, 255))


def enable_landmarks(filename, enable):
    descr, thumbnail = load_face_description(filename)
    enabled = thumbnail is not None

    if enable == enabled:
        logging.debug(f'enable_landmarks skip: {filename}')
        return

    if 'landmarks' not in descr:
        logging.warning(f'has no landmarks: {filename}')
        return

    image = Image.open(filename)
    if enable:
        thumbnail = image.copy()
        __set_landmarks_lines(image, descr['landmarks'])
    else:
        image = thumbnail
        thumbnail = None
    save_with_description(image, descr, thumbnail, filename)


def save_face(out_filename, image, box, encoding, landmarks, out_size):
    top, right, bottom, left = box
    d = (bottom - top) // 2
    top = max(0, top - d)
    left = max(0, left - d)
    bottom = bottom + d
    right = right + d

    out_image = image[top:bottom, left:right]

    im = Image.fromarray(out_image)
    im.thumbnail((out_size, out_size))

    face_landmarks = {}
    if landmarks:
        hk = im.size[0] / (right - left)
        vk = im.size[1] / (bottom - top)
        for landmark, pts in landmarks.items():
            face_pts = []
            for pt in pts:
                face_pts.append((
                    int((pt[0] - left) * hk),
                    int((pt[1] - top) * vk)))
            face_landmarks[landmark] = face_pts

    thumbnail = im.copy()
    __set_landmarks(im, face_landmarks)
    descr = {'encoding': encoding, 'landmarks': face_landmarks}

    save_with_description(im, descr, thumbnail, out_filename)


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
