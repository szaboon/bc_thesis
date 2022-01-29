"""
function YOLO:
 - we are using model yolov3 to recognize objects in the image
 - to display bounding boxes we use OpenCV 4.4.0
"""

import numpy as np
import cv2
import platform
import time
import os
import multiprocessing
from espeakng import ESpeakNG
from PIL import ImageFont, ImageDraw, Image

# importing functions from robohlava version.1
from camera import Camera
import config as conf

# Flags
yolo_say = True
show_depth_map = True
print_detected_objects = True
print_FPS = False


class YOLO:

    def __init__(self):
        self.width = conf.WIDTH
        self.height = conf.HEIGHT

        self.detected_objects = DetectedObjects()
        self.camera = Camera()

        self.blank_image = None
        self.rgb_original = None
        self.rgb_copy_cv = None
        self.depth = None
        self.depth_colorized = None

        self.TIMER = 0
        self.info = {}
        self.list_of_objects = []

        self.voice = ESpeakNG()
        self.voice.voice = 'czech'
        self.voice.pitch = 50
        self.voice.speed = 150
        self.talk_after = 4

        if __name__ == "__main__":
            if platform.system() == "Windows":
                self.base_dir = os.getcwd() + "\.."
            else:
                self.base_dir = os.getcwd() + "/.."

        print("[INFO] Base dir is ", self.base_dir)
        print("[INFO] Loading Yolo model...")

        if platform.system() == "Windows":
            """Define path and load YOLO object detection model"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'\models\yolo\yolov3.cfg')
            weights_path = os.path.join(self.base_dir +
                                        r'\models\yolo\yolov3.weights')

            self.yolo_net = cv2.dnn.readNetFromDarknet(prototxt_path, weights_path)
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        else:
            """Define path and load YOLO object detection model"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'/models/yolo/yolov3.cfg')
            weights_path = os.path.join(self.base_dir +
                                        r'/models/yolo/yolov3.weights')

            self.yolo_net = cv2.dnn.readNetFromDarknet(prototxt_path, weights_path)
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        print("[INFO] Yolo was loaded")

        # Defining font path, more options to choose in fonts repository
        self.fontpath = os.getcwd() + "/fonts/MontserratAlternates-Regular.otf"

    def obj_det_yolo(self):
        # Defining variables
        text = []
        classes = conf.classes_cz
        layer_names = self.yolo_net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        height, width, channels = self.rgb_original.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(self.rgb_original, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo_net.setInput(blob)
        outs = self.yolo_net.forward(output_layers)
        # Getting information from image
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Showing only objects with 50% probability
                if confidence > 0.5:
                    # Object size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = ImageFont.truetype(self.fontpath, 32)
        # Setting color and label for detected objects
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if yolo_say:    # YOLO saying detected objects
                    self.list_of_objects = self.queue(classes[class_ids[i]])
                    if len(self.list_of_objects) > self.talk_after:
                        print("Say objects ---> executed")
                        self.say_objects()
                for name_key, color_val in conf.color_dict_cz.items():
                    if name_key == label:
                        color_yolo = color_val
                        # Text
                        img_pil = Image.fromarray(self.rgb_copy_cv)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((x, y - 35), str(label), font=font, fill=(color_yolo[0],color_yolo[1],color_yolo[2],0))
                        self.rgb_copy_cv = np.array(img_pil)
                        # Rectangle
                        cv2.rectangle(self.rgb_copy_cv, (x, y), (x + w, y + h),
                                      color_yolo, 2)
                        obj_img = self.rgb_original[y:y + h, x:x + w]
                        self.detected_objects.append(label, boxes[i], confidences[i], obj_img)
        self.objects_yolo.append(text)
        return

    def queue(self, object):
        if object in self.list_of_objects:
            pass
        else:
            self.list_of_objects.insert(0, object)
            if print_detected_objects is True:
                print(self.list_of_objects)
        return self.list_of_objects

    def say_objects(self):
        string = ""
        for _ in range(len(self.list_of_objects)):
            string = string + self.list_of_objects.pop() + "..."
        p2 = multiprocessing.Process(target=self.voice.say, args=[string])
        p2.start()
        p2.join()

    def get_frames(self):
        self.rgb_original, self.depth, self.depth_colorized = self.camera.take_frame()
        self.rgb_copy_cv = np.copy(self.rgb_original)

    def show_frame(self):
        # Displaying all the information
        self.img_yolo = cv2.resize(self.rgb_copy_cv, None, fx=1.5, fy=1.5)  # Size of RGB image
        cv2.imshow("yolo.py", self.img_yolo)
        if show_depth_map is True:
            cv2.imshow("depth map", self.depth_colorized)

    def terminate(self):
        cv2.destroyAllWindows()
        self.camera.terminate()


class DetectedObjects:
    def __init__(self):
        self.objects_list = []

    def append(self, label, box, confidence, image):
        if label not in [obj.label for obj in self.objects_list]:
            self.objects_list.append(DetectedObject(label, box, confidence, image))
        else:
            for obj in self.objects_list:
                if obj.label == label:
                    obj.update(box, confidence, image)

    def clear(self):
        del self.objects_list
        self.objects_list = []


class DetectedObject:
    def __init__(self, label, box, confidence, image):
        self.label = label
        self.box = box
        self.confidence = confidence
        self.image = image

    def update(self, box, confidence, image):
        self.box = box
        self.confidence = confidence
        self.image = image

    def __del__(self):
        pass


if __name__ == "__main__":
    print("Modul  | {0} | ---> executed".format(__name__))
    image_process = YOLO()
    print("[INFO] Running object recognition\n\n"
          "Press 'Q' to stop recognition\n")
    while True:
        prev_time = time.time()
        image_process.objects_yolo = []
        image_process.get_frames()
        image_process.obj_det_yolo()
        image_process.show_frame()
        if print_FPS is True:
            print("FPS: ", round(1 / (time.time() - prev_time),2))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            image_process.camera.terminate()
            cv2.destroyAllWindows()
            break