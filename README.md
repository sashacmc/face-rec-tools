# Face Rec Tools
Media library face recognition tools

## Features:
  * Detect faces in photo/video files
  * Match faces to face patterns
  * Patterns manipulation
  * Face database search
  * Tagging faces in Plex Media Server library 

### Requirements:

  * Debian based Linux (Other Linux versions not officially supported, but might work)
  * Python 3.6+
  * NVIDIA GPU (optional)
  * Plex Media Server (optional)

### Dependencies:
  * OpenCV + python3 bindings (better with CUDA support)
  * DLib + python3 bindings (better with CUDA support)
  * TensorFlow + python3 binding (optional)
  * deepface python3 library (optional)
  * numpy python3 library
  * face_alignment python3 library
  * face_recognition python3 library
  * python3-piexif python3 library

## Installation

### Dependencies installation:

#### Build tools 
For building dlib and others
```bash
sudo apt-get install build-essential cmake
```

#### OpenCV (skip if you have NVIDIA GPU):
```bash
sudo apt-get install python3-opencv
```

#### CUDA, OpenCV and TensorFlow installation (skip if don't have NVIDIA GPU):
Installation of libraries with CUDA is not so easy and may be differ for different OS versions.
Some useful links for CUDA setup

https://developer.nvidia.com/cuda-downloads

https://medium.com/@sb.jaduniv/how-to-install-opencv-4-2-0-with-cuda-10-1-on-ubuntu-20-04-lts-focal-fossa-bdc034109df3

https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/

#### Deep Face library installation (skip if don't use deepface encoding, default):
```bash
pip3 install deepface
```

#### Face Recognition library
Install my face_recognition clone (there are some modification to support exteranal matched faces)
```bash
git clone https://github.com/sashacmc/face_recognition.git; cd face_recognition; pip3 install .; cd ..
```

### Face Rec Tools installation
```bash
git clone https://github.com/sashacmc/face-rec-tools.git; cd face-rec-tools; pip3 install .; cd ..
```

## Usage

### Config file
Use one from predefined config files or prepare you own.

Predefined config files located in package's cfg folder (e.g. ~/.local/lib/python3.6/site-packages/face_rec_tools/cfg/)

Copy it to default config location (~/.face-rec.cfg) or specify with command line switch "-c"
```bash
cp ~/.local/lib/python3.?/site-packages/face_rec_tools/cfg/frontal.cfg ~/.face-rec.cfg
```

### Command-Line Interface

```bash
face-rec-cli
```

### Web Interface

```bash
face-rec-server 
```

### Plex Media Server synchronisation

```bash
face-rec-plexsync
```
