#!/usr/bin/python3

import sys
import pickle
import piexif
import face_recognition

import tools
import faceencoder


encoder = faceencoder.FaceEncoder('VGG-Face', 'cosine')


def get_face(fname):
    try:
        encoding = pickle.loads(
            piexif.load(fname)["0th"][piexif.ImageIFD.ImageDescription])
        print('Use cached: ' + fname)
        return encoding
    except Exception:
        pass

    image = tools.read_image(fname, 1000)
    boxes = face_recognition.face_locations(image, model='cnn')
    return encoder.encode(image, boxes)[0]


def main():
    encoding_base = get_face(sys.argv[1])
    fnames = sys.argv[2:]
    encodings = [get_face(fname) for fname in fnames]

    distances = encoder.distance(encodings, encoding_base)

    dist_name = [(distances[i], fname) for i, fname in enumerate(fnames)]
    dist_name.sort()

    for d, n in dist_name:
        print(d, '\t', n)


if __name__ == '__main__':
    main()
