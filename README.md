# Face Rec Tools
Media library face recognition tools

## Features
  * Detect faces in photo/video files
  * Match faces to face patterns
  * Patterns manipulation
  * Face database search
  * Tagging faces in Plex Media Server library 

## Requirements
  * Debian based Linux (Other Linux versions not officially supported, but might work)
  * Python 3.6+
  * NVIDIA GPU (optional)
  * Plex Media Server (optional)

## Dependencies
  * OpenCV + python3 bindings (better with CUDA support)
  * DLib + python3 bindings (better with CUDA support)
  * TensorFlow + python3 binding (optional)
  * deepface python3 library (optional)
  * numpy python3 library
  * face_alignment python3 library
  * face_recognition python3 library
  * python3-piexif python3 library

## Installation

### Dependencies installation

#### Build tools and install tools
For building dlib and others
```bash
sudo apt-get install build-essential cmake python3-pip
```

#### OpenCV (skip if you have NVIDIA GPU)
```bash
sudo apt-get install python3-opencv
```

#### CUDA, OpenCV and TensorFlow installation (skip if don't have NVIDIA GPU)
Installation of libraries with CUDA is not so easy and may be differ for different OS versions.
Some useful links for CUDA setup

https://developer.nvidia.com/cuda-downloads

https://medium.com/@sb.jaduniv/how-to-install-opencv-4-2-0-with-cuda-10-1-on-ubuntu-20-04-lts-focal-fossa-bdc034109df3

https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/

#### Deep Face library installation (skip if don't use deepface encoding, default)
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
# recognize single image and print output (useful for debug)
face-rec-cli -a recognize_image -i imagefile.jpg

# recognize single video and print summary output (useful for debug)
face-rec-cli -a recognize_video -i videofile.mp4

# recognize folder and store the result in the database
face-rec-cli -a recognize_folder -i /folder/with/images

# remove folder recognition the result from the database
face-rec-cli -a remove_folder -i /folder/with/images

# match/rematch unmatched faces in database
face-rec-cli -a match_unmatched

# match/rematch all faces in database
face-rec-cli -a match_all

# match/rematch faces from folder in database
face-rec-cli -a match_folder -i /folder/with/images

# save faces from folder (must be previously recognized)
# cachedb must be disabled (otherwise they will saved inside cachedb)
face-rec-cli -a save_faces -i /folder/with/images -o /folder/for/faces

# find face from image in database and save them in folder
# cachedb must be disabled (otherwise they will saved inside cachedb)
face-rec-cli -a get_faces_by_face -i image_with_one_face.jpg -o /folder/for/faces
```

### Web Interface
Run the server from the command line
```bash
face-rec-server 
```
Open the browser with you hostname/ip_adress and port 8081

For recognize a new folder click to "Recognition"->"Add new files...".
And specify a folder which you want to recognize.
![face_rec_screen_1](https://user-images.githubusercontent.com/28735879/104759965-fa22e580-5760-11eb-9e18-e20cc340c96f.png)

First time recognition will take a while, because of loading necessary models.

After first recognition you will get all faces as unknown.
![face_rec_screen_3](https://user-images.githubusercontent.com/28735879/104760428-a1a01800-5761-11eb-9765-cf036d2639f7.png)

Click to the face (or select several with Shift/Ctrl) to add it as a pattern.
![face_rec_screen_4](https://user-images.githubusercontent.com/28735879/104760644-edeb5800-5761-11eb-9e92-e91c159e28d6.png)

Previously added names will be saved, and you don't need to input it again.
If you want to change the default face logo you can add it as 0_face.jpg for each person subfolder.
![face_rec_screen_5](https://user-images.githubusercontent.com/28735879/104760969-75d16200-5762-11eb-8e8e-0cdc55f38eb3.png)

After patterns adding, start the match again (e.g. "Match"->"Rematch folder...")
![face_rec_screen_6](https://user-images.githubusercontent.com/28735879/104761100-ad400e80-5762-11eb-96eb-616d1dd969f3.png)

After matching with patterns you will have matched persons and "weak" matched persons, it means that it not fully matched and will not be used for sync or search.
You need to check them and add to patterns.
![face_rec_screen_7](https://user-images.githubusercontent.com/28735879/104761475-3f481700-5763-11eb-843d-4e59fc49e97a.png)

If you need some additional info about the face you can click the ![srclink](https://user-images.githubusercontent.com/28735879/104761676-8c2bed80-5763-11eb-8d53-bae7abd7573c.png) icon to see the source file, or the ![pattlink](https://user-images.githubusercontent.com/28735879/104761761-a239ae00-5763-11eb-9f5a-9ec4d7189ee6.png) to see the pattern which it has been matched to.

If you have a big amount of faces you can simplify patterns separation by mean of clusterization in menu Clustering it will separate near faces to groups and will allow to add it to patterns together.

### Plex Media Server synchronisation
Face Recognition Tools allow syncing recognition results with Plex Media Server by means of tags.
Recognized files will tagged with tags "person:[PERSON_NAME]"

```bash
# set all tags to the Plex database 
face-rec-plexsync -a set_tags

# clear all tags from the Plex database
face-rec-plexsync -a remove_tags

# recognize all files which present in the Plex database
# but not recognized yet and store the result in the database 
face-rec-plexsync -a sync_new

# delete from database all files which not present in the Plex database
face-rec-plexsync -a sync_deleted
```

### Seach in DB without Plex
If you have no Plex, or want to use some more complex search you can use follow command 

(it will search all files in folder 2020 which contains faces of persons Name1 and Name2)

```bash
face-rec-db -a find_files_by_names -f 2020 -n Name1,Name2
```
