#!/usr/bin/python3

import sys
import face_recognition

import tools
import faceencoder


encoder = faceencoder.FaceEncoder('VGG-Face', 'cosine', align=True)


def get_face(fname):
    """
    try:
        encoding = pickle.loads(
            piexif.load(fname)["0th"][piexif.ImageIFD.ImageDescription])
        print('Use cached: ' + fname)
        return encoding['encoding']
    except Exception:
        pass
"""
    try:
        image = tools.read_image(fname, 1000)
    except Exception:
        print(f'image {fname} reading failed')
        return None
    boxes = face_recognition.face_locations(image, model='cnn')
    if len(boxes) != 1:
        print(f'Image contains {len(boxes)} faces')
        return None
    return encoder.encode(image, boxes)[0][0]


def main():
    encoding_base = get_face(sys.argv[1])
    if encoding_base is None:
        return
    fnames = sys.argv[2:]
    encodings = [get_face(fname) for fname in fnames]

    encodings_filtered = []
    fnames_filtered = []
    for e, f in zip(encodings, fnames):
        if e is not None:
            encodings_filtered.append(e)
            fnames_filtered.append(f)

    distances = encoder.distance(encodings_filtered, encoding_base)

    dist_name = [(distances[i], fname)
                 for i, fname in enumerate(fnames_filtered)]
    dist_name.sort()

    for d, n in dist_name:
        print(d, '\t', n)


if __name__ == '__main__':
    main()
