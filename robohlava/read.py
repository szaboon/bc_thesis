"""
function READ
 - reads text from image and prints it in the terminal
"""

import os
import numpy as np
import cv2
import pytesseract
import re
import platform
from PIL import Image
import time

# importing functions from robohlava version.1
from camera import Camera
import config as conf

FPS_flag = False


class ReadTesseract:
    """Class for reading using tesseract"""

    def __init__(self):
        self.width = conf.WIDTH
        self.height = conf.HEIGHT

        self.camera = Camera()

        self.final_image = None
        self.book_text = []
        self.text = ""

        if __name__ == "__main__":
            if platform.system() == "Windows":
                self.base_dir = os.getcwd() + "\\.."
            else:
                self.base_dir = os.getcwd() + "/.."
        else:
            self.base_dir = os.getcwd()  # + "\\.."

        print("[INFO] Base dir is ", self.base_dir)

    def book_read(self):
        self.get_frames()
        text = pytesseract.image_to_string(Image.fromarray(self.rgb_original), lang="ces")
        self.text = re.sub('[^.,:!?\w]', ' ', text)
        print(self.text)

    def get_frames(self):
        self.rgb_original, self.depth, self.depth_colorized = self.camera.take_frame()
        self.rgb_copy_cv = np.copy(self.rgb_original)

    def show_frame(self):
        self.imgage = cv2.resize(self.rgb_copy_cv, None, fx=1.5, fy=1.5)  # Size of RGB image
        cv2.imshow("read.py", self.imgage)

    def draw_book_rectangle(self):
        cv2.rectangle(self.rgb_copy_cv, (50, 100), (int(self.width - 50),
                                                    int(self.height - 30)), (255, 255, 255), 2)

    def terminate(self):
        cv2.destroyAllWindows()
        self.camera.terminate()


if __name__ == "__main__":

    print("Modul  | {0} | ---> executed".format(__name__))
    read = ReadTesseract()
    while True:
        prev_time = time.time()
        read.book_read()
        read.draw_book_rectangle()
        read.show_frame()
        if FPS_flag is True:
            print("FPS: ", round(1 / (time.time() - prev_time), 2))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            ReadTesseract.camera.terminate()
            cv2.destroyAllWindows()
            break
