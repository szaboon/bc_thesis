# ROBOHLAVA

Here are all the things you need to run robotic head except models for object and face detection.
# Installation

1.  Linux os (Ubuntu 18.04 or other versions)
2.  Install Cuda: \
    Download -  <https://developer.nvidia.com/cuda-toolkit-archive> \
    Installation guide - <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html> \
3.  Install cuDNN: \
    Download - <https://developer.nvidia.com/rdp/cudnn-download> \
    Installation guide - <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html> \
    If you feel lost - <https://www.youtube.com/watch?v=fyHbV3XhBoM&ab_channel=CodeEnjoy>
4.  Install tesseract (czech) - <https://github.com/tesseract-ocr/tesseract>
5.  Download OpenCV - <https://opencv.org/releases/> \
    and opencv-contrib (versions in branches) - <https://github.com/opencv/opencv_contrib/tree/4.4.0>
6.  Install Opencv with CUDA, cuDNN and tesseract using cmake: \
    Installation guide - <https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7> \
7.  Create virtual environment with python 3.7 (shown in step.5)
8.  Install Intel RealSense SDK 2.0 - <https://www.intelrealsense.com/sdk-2/> \
    Install python library in virtual environment:
    
        pip3 install pyrealsense2
    
9.  Install other necessary libraries shown in each function 
<!-- end list -->

# Downloading pre-trained models
These models should be in a file called "models", in robohlava directory.
1.  yolov3 model

        wget https://pjreddie.com/media/files/yolov3.weights

2.  face detection caffemodel - <https://github.com/vinuvish/Face-detection-with-OpenCV-and-deep-learning/tree/master/models> \
3.  age and gender detection caffemodels - <https://gist.github.com/GilLevi/c9e99062283c719c03de>


# Files

| Directory/File  | Description                              |
| --------------- | ---------------------------------------- |
| arduino\_files  | Arduino code and examples                |
| fonts           | Fonts used with OpenCV (yolo.py)         |
| camera.py       | Basic camera functions (RealSense)       |
| person_class.py | Person tracking algorithm                |
| config.py       | Configuration file                       |
| yolo.py         | function for objects detection           |
| jarvis.py       | function for answering questions         |
| track.py        | function for person tracking             |
| read.py         | function for reading                     |
| repeat.py       | function for repeating your words (cz,sk)|

# Contacts

  - Author: Michal Szabó
  - Email: szmiso1999@gmail.com
  - Date: 2021, Brno