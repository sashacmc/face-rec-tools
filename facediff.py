#!/usr/bin/python3

import sys
import face_recognition

import tools


def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    image1 = tools.read_image(file1, 1000)
    image2 = tools.read_image(file2, 1000)

    boxes1 = face_recognition.face_locations(image1, model='cnn')
    encodings1 = face_recognition.face_encodings(image1, boxes1, 1)

    boxes2 = face_recognition.face_locations(image2, model='cnn')
    encodings2 = face_recognition.face_encodings(image2, boxes2, 1)

    distances = face_recognition.face_distance(encodings1, encodings2[0])

    print(distances)


if __name__ == '__main__':
    main()
