# Face-Rec-Tools
Media library face recognition tools

## Features:
  * Detect faces in photo/video files
  * Match faces to face patterns
  * Patterns manipulation
  * Face database search
  * Tagging faces in Plex library 

### Requirements:

  * Python 3.6+
  * Debian based Linux (Other Linux versions not officially supported, but might work)
  * NVIDIA GPU (optional)

### Dependencies:
  * OpenCV + python3 bindings (better with CUDA support)
  * DLib + python3 bindings
  * TensorFlow + python3 bindings
  * numpy python3 library
  * deepface python3 library
  * face_alignment python3 library
  * face_recognition python3 library
  * python3-piexif python3 library

## Installation

### Dependencies installation:
#### CUDA installation:
https://developer.nvidia.com/cuda-downloads

#### TensorFlow installation: 
https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/

#### OpenCV installation:
https://medium.com/@sb.jaduniv/how-to-install-opencv-4-2-0-with-cuda-10-1-on-ubuntu-20-04-lts-focal-fossa-bdc034109df3

#### Python libraries installation:
(some of libraries was already installed with GPU support, please skip it if necessary)
```bash
pip3 install numpy 
pip3 install opencv-python
pip3 install dlib 
pip3 install piexif
pip3 install pillow
pip3 install face_alignment
```
Install my face_recognition clone (there are some modification to support exteranal matched faces)
```bash
git clone https://github.com/sashacmc/face_recognition.git; cd face_recognition; pip3 install .;
```

### Face-Rec-Tools installation
```bash
git clone https://github.com/sashacmc/face-rec-tools.git
cd face-rec-tools
pip3 install .
```
## Usage

### Config file
Use one from predefined config files or prepare you own.

Predefined config files located in package's cfg folder (e.g. ~/.local/lib/python3.6/site-packages/face_rec_tools/cfg/)

Copy it to default config location (~/.face-rec.cfg) or specify with command line switch "-c"
```bash
cp ~/.local/lib/python3.6/site-packages/face_rec_tools/cfg/frontal.cfg ~/.face-rec.cfg
```

### Command-Line Interface

```bash
face-rec-cli
```

### Web Interface

```bash
face-rec-server 
```
