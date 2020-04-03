import io
import os
import cv2
import sys
import glob
import math
import piexif
import pickle
import logging
import collections
from PIL import Image, ImageDraw


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
VIDEO_EXTS = ('.mp4', '.mpg', '.mpeg', '.mov', '.avi', '.mts')


def seconds_to_str(s):
    hour = int(s / 3600)
    mins = int((s - hour * 3600) / 60)
    secs = int(s - hour * 3600 - mins * 60)
    return f'{hour}:{mins:02}:{secs:02}'


def __list_files(path, exts, nomedia_names):
    files = []
    dirs = []
    if os.path.isfile(path):
        return (path,)
    for e in os.scandir(path):
        if e.is_file():
            if e.name in nomedia_names:
                return []
            ext = get_low_ext(e.name)
            if exts is None or ext in exts:
                files.append(e.path)
        elif e.is_dir():
            dirs.append(e.path)
    for d in dirs:
        files += __list_files(d, exts, nomedia_names)
    return files


def list_files(path, exts=None, nomedia_names=()):
    logging.debug(f'list_files: {path}, {exts}, {nomedia_names}')
    files = []
    for p in glob.glob(path):
        files += __list_files(p, exts, nomedia_names)
    return files


def cuda_init(tf_memory_limit=1536):
    import tensorflow as tf

    logging.debug('cuda init')
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    if tf.config.experimental.get_memory_growth(gpu):
        return
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(
            memory_limit=tf_memory_limit)])


def cuda_release():
    import torch

    logging.debug('cuda release')
    torch.cuda.empty_cache()


def read_image(image_file, max_size):
    image = cv2.imread(image_file)
    return prepare_image(image, max_size)


def read_video(video_file, max_size, max_video_frames):
    video = cv2.VideoCapture(video_file)
    frames = []
    ret = True
    while ret and max_video_frames != 0:
        max_video_frames -= 1
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


def bound_size(line):
    xmin, ymin = line[0]
    xmax, ymax = line[0]
    for x, y, in line[1:]:
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    size = math.hypot(xmax - xmin, ymax - ymin)
    return size


def calc_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) -
                       math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang


def test_line_angle(line):
    for i in range(len(line) - 3):
        angle = abs(calc_angle(line[i], line[i + 1], line[i + 2]))
        if angle < 45:
            return False
    return True


def test_landmarks(l):
    if l is None:
        logging.debug('Empty landmarks')
        return False
    if 'chin' not in l:
        logging.debug('landmarks without chin')
        return False
    size = bound_size(l['chin'])
    if not test_line_angle(l['chin']):
        logging.debug('landmarks chin angle test failed')
        return False
    if bound_size(l['left_eye']) >= size / 4 or \
       bound_size(l['right_eye']) >= size / 4:
        logging.debug('landmarks eye size test failed')
        return False
    if bound_size(l['left_eyebrow']) >= size / 2 or \
       bound_size(l['right_eyebrow']) >= size / 2:
        logging.debug('landmarks eyebrow size test failed')
        return False
    if bound_size(l['nose_tip']) >= size / 4 or \
       bound_size(l['nose_bridge']) >= size / 2:
        logging.debug('landmarks nose size test failed')
        return False
    return True


def filter_encoded_faces(encoded_faces):
    return list(filter(
        lambda enc: test_landmarks(enc['landmarks']), encoded_faces))


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
             'face_id': enc['face_id'],
             'src': src_filename}

    save_with_description(im, descr, thumbnail, out_filename)


def filter_images(files_faces):
    return list(filter(
        lambda ff: get_low_ext(ff['filename']) in IMAGE_EXTS,
        files_faces))


def reduce_faces_from_video(faces, min_count):
    def test_face(face):
        return face['name'] != '' and \
            (face['count'] > min_count or face['dist'] < 0.01)

    dct = collections.defaultdict(lambda: {'dist': sys.maxsize, 'count': 0})
    for face in faces:
        name = face['name']
        count = dct[name]['count'] + 1
        if face['dist'] < dct[name]['dist']:
            dct[name] = face
        dct[name]['count'] = count
    res = []
    for face in dct.values():
        ok = True
        if face['name'].endswith('_weak'):
            n = face['name'][:-5]
            if n in dct:
                if test_face(dct[n]):
                    # skip weak because of name already persist in current file
                    ok = False
                    logging.debug(f'skip weak: {n}')
                else:
                    # extend weak count by already persisted name
                    face['count'] += dct[n]['count']
                    logging.debug(f'extend weak: {n}')
        if ok:
            ok = test_face(face)
        logging.debug(f'faces in video: {face["name"]}: {face["count"]}: {ok}')
        if ok:
            res.append(face)
    return res


def reduce_faces_from_videos(files_faces, min_count):
    res = []
    for ff in files_faces:
        if get_low_ext(ff['filename']) in VIDEO_EXTS:
            ff['faces'] = reduce_faces_from_video(ff['faces'], min_count)
        res.append(ff)
    return res


def get_low_ext(filename):
    return os.path.splitext(filename)[1].lower()


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
    def __init__(self, video_file, max_size, max_video_frames):
        self.__video_file = video_file
        self.__max_size = max_size
        self.__max_video_frames = max_video_frames
        self.__frames = None

    def frames(self):
        if self.__frames is None:
            logging.debug(f'LazyVideo load: {self.__video_file}')
            self.__frames = read_video(self.__video_file,
                                       self.__max_size,
                                       self.__max_video_frames)
        return self.__frames

    def get(self, frame_num):
        return self.frames()[frame_num]

    def filename(self):
        return self.__video_file


def load_media(media_file, max_size, max_video_frames):
    ext = get_low_ext(media_file)
    if ext in IMAGE_EXTS:
        return LazyImage(media_file, max_size)
    elif ext in VIDEO_EXTS:
        return LazyVideo(media_file, max_size, max_video_frames)
    else:
        raise Exception(f'Unknown ext: {ext}')
