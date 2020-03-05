import io
import os
import cv2
import piexif
import pickle
import logging
from PIL import Image, ImageDraw

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
VIDEO_EXTS = ('.mp4', '.mpg', '.mpeg', '.mov', '.avi', '.mts')


def read_image(image_file, max_size):
    image = cv2.imread(image_file)
    return prepare_image(image, max_size)


def read_video(video_file, max_size):
    video = cv2.VideoCapture(video_file)
    frames = []
    ret = True
    while ret:
        ret, frame = video.read()
        if ret:
            frame = prepare_image(frame, max_size)
            frames.append(frame)
    return frames


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


def load_face_thumbnail(filename):
    exif = piexif.load(filename)
    return exif.pop('thumbnail')


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
    if len(face_landmarks) == 0:
        return
    draw = ImageDraw.Draw(image)
    for pts in face_landmarks.values():
        draw.line(pts, fill=(255, 255, 255))


def enable_landmarks(filename, enable):
    descr, thumbnail = load_face_description(filename)
    enabled = thumbnail is not None

    if enable == enabled:
        logging.debug(f'enable_landmarks skip: {filename}')
        return

    if descr is None or 'landmarks' not in descr:
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


def save_face(out_filename, image, enc, out_size, src_filename):
    top, right, bottom, left = enc['box']
    d = (bottom - top) // 2
    top -= d
    left -= d
    bottom += d
    right += d

    height, width = image.shape[0:2]
    out_image = image[max(0, top):bottom, max(0, left):right]
    if top < 0 or left < 0 or right > width or bottom > height:
        out_image = cv2.copyMakeBorder(out_image,
                                       max(0, -top), max(0, bottom - height),
                                       max(0, -left), max(0, right - width),
                                       cv2.BORDER_CONSTANT, None, 0)

    im = Image.fromarray(out_image)
    im.thumbnail((out_size, out_size))

    face_landmarks = {}
    if 'landmarks' in enc and enc['landmarks']:
        hk = im.size[0] / (right - left)
        vk = im.size[1] / (bottom - top)
        for landmark, pts in enc['landmarks'].items():
            face_pts = []
            for pt in pts:
                face_pts.append((
                    int((pt[0] - left) * hk),
                    int((pt[1] - top) * vk)))
            face_landmarks[landmark] = face_pts

    thumbnail = im.copy()
    __set_landmarks_lines(im, face_landmarks)
    descr = {'encoding': enc['encoding'],
             'landmarks': face_landmarks,
             'box': enc['box'],
             'frame': enc['frame'],
             'src': src_filename}

    save_with_description(im, descr, thumbnail, out_filename)


class LazyImage(object):
    def __init__(self, image_file, max_size):
        self.__image_file = image_file
        self.__max_size = max_size
        self.__image = None

    def get(self, dummy_frame_num=0):
        if self.__image is None:
            logging.debug(f'LazyImage load: {self.__image_file}')
            self.__image = read_image(self.__image_file, self.__max_size)
        return self.__image

    def filename(self):
        return self.__image_file


class LazyVideo(object):
    def __init__(self, video_file, max_size):
        self.__video_file = video_file
        self.__max_size = max_size
        self.__frames = None

    def frames(self):
        if self.__frames is None:
            logging.debug(f'LazyVideo load: {self.__video_file}')
            self.__frames = read_video(self.__video_file, self.__max_size)
        return self.__frames

    def get(self, frame_num):
        return self.frames()[frame_num]

    def filename(self):
        return self.__video_file


def load_media(media_file, max_size):
    ext = os.path.splitext(media_file)[1].lower()
    if ext in IMAGE_EXTS:
        return LazyImage(media_file, max_size)
    elif ext in VIDEO_EXTS:
        return LazyVideo(media_file, max_size)
    else:
        raise Exception(f'Unknown ext: {ext}')
