"""
WHEN USING ARDUINO:                                       initial position               normal position
    Set the initial position of the robotic head:             mot.2
     - camera should be pointed straight up                -- mot.1    arduino            -- mot.1 mot2 arduino
     - motor 2 shoud be in the back (initial position)        front                          front
     - camera is above mot.1
    When you run this function, motors will move from initial position to normal position
    and camera will be facing person in front of robohlava.
"""

import os
import numpy as np
import cv2
import platform
import time
import serial
import config as conf

# importing functions from robohlava version.1
from camera import Camera
from person_class import Persons_class

FPS_flag = False
Arduino_track = False

class ImageProcessing:
    """Class for real-time person tracking
    Functions:
     - face recognition
     - age recognition
     - gender recognition
    You can also use yolo object detection but it's gonna be slower
    """

    def __init__(self):
        self.width = conf.WIDTH
        self.height = conf.HEIGHT

        self.centroid_to_arduino = []
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1);
        time.sleep(5)
        self.start_byte = 55
        self.stop_byte = 110
        self.movex = 30
        self.movey = 20
        self.motor_change = 1
        self.lag = 0
        self.last_centroid = (320, 240)

        self.camera = Camera()

        self.PersonsObjects = None

        self.blank_image = None
        self.rgb_original = None
        self.rgb_copy_cv = None
        self.depth = None
        self.depth_colorized = None
        self.img_mini = None

        self.final_image = None
        self.book_text = None

        self.img_mini_persons_objects = None

        self.TIMER = 0
        self.info = {}

        if __name__ == "__main__":
            if platform.system() == "Windows":
                self.base_dir = os.getcwd() + "\\.."
            else:
                self.base_dir = os.getcwd() + "/.."
        else:
            self.base_dir = os.getcwd()  # + "\\.."

        print("[INFO] Base dir is ", self.base_dir)

        print("[INFO] Starting load models ...")

        if platform.system() == "Windows":

            """Define path and load face_detector"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'\models\face_detector\deploy.prototxt')
            weights_path = os.path.join(self.base_dir +
                                        r'\models\face_detector\weights.caffemodel')

            self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
            self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            """Define path and load age detector"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'\models\age_detector\age_deploy.prototxt')
            weights_path = os.path.join(self.base_dir +
                                        r'\models\age_detector\age_net.caffemodel')

            self.age_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
            self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            """Define path and load gender detector"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'\models\gender_detector\deploy_gender.prototxt')
            weights_path = os.path.join(self.base_dir +
                                        r'\models\gender_detector\gender_net.caffemodel')

            self.gender_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
            self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        else:

            """Define path and load face_detector"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'/models/face_detector/deploy.prototxt')
            weights_path = os.path.join(self.base_dir +
                                        r'/models/face_detector/weights.caffemodel')

            self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
            self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            """Define path and load age detector"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'/models/age_detector/age_deploy.prototxt')
            weights_path = os.path.join(self.base_dir +
                                        r'/models/age_detector/age_net.caffemodel')

            self.age_net = cv2.dnn.readNet(prototxt_path, weights_path)
            self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            """Define path and load gender detector"""
            prototxt_path = os.path.join(self.base_dir +
                                         r'/models/gender_detector/deploy_gender.prototxt')
            weights_path = os.path.join(self.base_dir +
                                        r'/models/gender_detector/gender_net.caffemodel')

            self.gender_net = cv2.dnn.readNet(prototxt_path, weights_path)
            self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        print("[INFO] Models were loaded")

        """Person class + tracker algorithm"""
        self.tracker = Persons_class()

    def get_frames(self):
        self.rgb_original, self.depth, self.depth_colorized = self.camera.take_frame()
        self.rgb_copy_cv = np.copy(self.rgb_original)

    def centroid(self, persons_data_collection):
        self.PersonsObjects = self.tracker.update(persons_data_collection)

    def person_tracking(self, change_person=False):
        if len(list(self.PersonsObjects.values())) > 0:
            for person in list(self.PersonsObjects.values()):
                if person.tracking_person:
                    if change_person:
                        person.tracking_person = False
                    else:
                        return person
            max_area_index = np.argmax([person.area for person in list(self.PersonsObjects.values())])
            list(self.PersonsObjects.values())[max_area_index].tracking_person = True
            return list(self.PersonsObjects.values())[max_area_index]

    def face_age_gender_detector(self, age_flag=False, gender_flag=False):
        AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
                       "(38-43)", "(48-53)", "(60-100)"]
        GENDER_BUCKETS = ["Male", "Female"]
        persons_data_collection = []
        model_mean_values = (78, 87, 114)

        (h, w) = self.rgb_original.shape[:2]

        face_blob = cv2.dnn.blobFromImage(cv2.resize(self.rgb_original,
                                                     (300, 300)), 1.0, (300, 300), model_mean_values)
        self.face_net.setInput(face_blob)
        detections = self.face_net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if self.rgb_original is not None:
                face_img = np.copy(self.rgb_original[startY:endY, startX:endX])
            else:
                continue
            if face_img.shape[0] < 15 or face_img.shape[1] < 15:
                continue

            try:
                face_img = cv2.resize(face_img, (227, 227), interpolation=cv2.INTER_CUBIC)
            except:
                continue

            if age_flag:
                age_blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227),
                                                 model_mean_values,
                                                 swapRB=True)
                self.age_net.setInput(age_blob)
                predictions = self.age_net.forward()

                i = predictions[0].argmax()
                age = AGE_BUCKETS[i]
                age_confidence = predictions[0][i]
            else:
                age = None
                age_confidence = None

            if gender_flag:
                gender_blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227),
                                                    model_mean_values, swapRB=False)
                self.gender_net.setInput(gender_blob)
                gender_preds = self.gender_net.forward()
                gender = GENDER_BUCKETS[gender_preds[0].argmax()]
            else:
                gender = None

            d = {
                "box": (startX, startY, endX, endY),
                "age": (age, age_confidence),
                "gender": gender,
                "img": np.asarray(face_img)
            }
            persons_data_collection.append(d)

        self.centroid(persons_data_collection)

    def persons_draw(self):
        i = 0
        for person in list(self.PersonsObjects.values()):
            i += 1
            text = "ID: "
            if i == 1:
                self.centroid_to_arduino = person.centroid
            for data in person.print_data:
                text += data + " "
            if person.tracking_person:
                color = conf.color_tracking_person
                text_tracking = "Tracking person (" + str(self.distance(person.centroid)) + "m)"
            else:
                color = conf.color_person
                text_tracking = " "

            cv2.rectangle(self.rgb_copy_cv, (person.box[0], person.box[1]),
                          (person.box[2], person.box[3]), color, 2)
            cv2.circle(self.rgb_copy_cv, tuple(person.centroid), 2, color)
            cv2.putText(self.rgb_copy_cv, text, (person.box[0], person.box[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            cv2.putText(self.rgb_copy_cv, text_tracking, (person.box[0], person.box[1] - 35),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    def distance_process(self, box):
        (xmin_depth, ymin_depth, xmax_depth, ymax_depth) = box

        depth = self.depth[xmin_depth:xmax_depth,
                ymin_depth:ymax_depth].astype(float)
        depth_scale = self.camera.profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
        dist, _, _, _ = cv2.mean(depth)
        return dist

    def distance(self, centre_point):
        distance = 0
        centre = [[centre_point[1], centre_point[0]], [centre_point[1]+5, centre_point[0]+5], [centre_point[1]+5,
                  centre_point[0]-5], [centre_point[1]-5, centre_point[0]+5], [centre_point[1]-5, centre_point[0]-5]]
        for average in centre:
            try:
                distance = distance + self.depth[average[0], average[1]]
            except IndexError:
                pass
        avg_distance = round(distance/5000, 2)
        return avg_distance

    def show_frame(self):
        self.imgage = cv2.resize(self.rgb_copy_cv, None, fx=1.5, fy=1.5)  # Size of RGB image
        cv2.imshow("track.py", self.imgage)
        cv2.imshow("depth map", self.depth_colorized)

    def draw_book_rectangle(self):
        cv2.rectangle(self.rgb_copy_cv, (50, 100), (int(self.width - 50),
                                                    int(self.height - 30)), (0, 0, 139), 2)

    def return_image(self):
        return self.rgb_copy_cv

    def terminate(self):
        cv2.destroyAllWindows()
        self.camera.terminate()

    def show_init_window(self):
        if platform.system == "Windows":
            self.final_image = cv2.imread(self.base_dir + r"\images\looser.png")
        else:
            self.final_image = cv2.imread(self.base_dir + r"/images/looser.png")
        self.final_image = cv2.resize(self.final_image, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    def update(self, flags):
        self.get_frames()

        if flags["track"]:
            self.face_age_gender_detector(flags["img_age"], flags["img_gender"])
            self.persons_draw()
            if flags["change_person"]:
                self.person_tracking(change_person=True)
            else:
                self.person_tracking(change_person=False)
            persons = list(self.PersonsObjects.values())
        else:
            persons = []

        self.final_image = np.hstack((self.rgb_copy_cv, self.depth_colorized))

        return (self.rgb_copy_cv, self.depth_colorized, persons)

    def angle2payload(self, angle):
        if angle >= 256:
            return [1, angle-256]
        else:
            return [0, angle]

    def track_person(self):
        if self.centroid_to_arduino is not None:
            self.calculate_movement()

    def calculate_movement(self):
        """__________        y (mot.2)
          |         |        0
          | 640x480 |    60--+--0 x (mot.1)
          |_________|       40
        """
        # Checking if the centroid has the same value (if its stuck)
        self.lag += 1
        if self.lag == 1:
            self.last_centroid = self.centroid_to_arduino
        if self.lag == 4:
            if self.centroid_to_arduino[0] == self.last_centroid[0]:
                self.centroid_to_arduino = [320, 240]
            self.lag = 0
        if self.lag > 4:
            self.lag = 0

        # Calculating movement if person is not in the middle of image
        try:
            if self.centroid_to_arduino[0] < 280:
                self.movex -= 1
                if self.movex < 0:
                    self.movex = 0
            if self.centroid_to_arduino[0] > 360:
                self.movex += 1
                if self.movex > 55:
                    self.movex = 55
            if self.centroid_to_arduino[1] < 200:
                self.movey -= 1
                if self.movey < 5:
                    self.movey = 5
            if self.centroid_to_arduino[1] > 280:
                self.movey += 1
                if self.movey > 30:
                    self.movey = 30
        except IndexError:
            pass
        self.move_motors()

    # You are not able to send msg to both motors at the same time, here si the solution to switch between them
    def move_motors(self):
        if self.motor_change == 1:
            self.move_x()
            self.motor_change = 2
        else:
            self.move_y()
            self.motor_change = 1

    def move_x(self):
        payload = [1, 1] + self.angle2payload(self.movex)
        msg = bytearray([self.start_byte, len(payload) + 3]) + bytearray(payload) + bytearray([self.stop_byte])
        self.ser.write(msg)

    def move_y(self):
        payload = [1, 2] + self.angle2payload(self.movey)
        msg = bytearray([self.start_byte, len(payload) + 3]) + bytearray(payload) + bytearray([self.stop_byte])
        self.ser.write(msg)


if __name__ == "__main__":
    print("Modul  | {0} | ---> executed".format(__name__))
    image_process = ImageProcessing()
    flags = {"img_arduino_track": True,
             "track": True,
             "img_age": True,
             "img_gender": True,
             "change_person": False,
             }
    counter = 0
    image_process.move_motors()
    time.sleep(0.5)
    image_process.move_motors()
    while True:
        prev_time = time.time()
        image_process.update(flags)
        image_process.show_frame()
        FPS = round(1 / (time.time() - prev_time), 2)
        if counter < 3:     # Every third frame is gonna be send to track (otherwise it crashes)
            counter += 1
        else:
            counter = 0
            image_process.track_person()
        if FPS_flag is True:
            print(FPS)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            image_process.camera.terminate()
            cv2.destroyAllWindows()
            break

else:
    print("Modul  | {0} | ---> imported".format(__name__))
