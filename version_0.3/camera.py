import cv2
from time import sleep

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise Exception("無法開啟攝像頭")
        self.frame = None
    def read(self):
        ret, frame = self.cap.read()

        if ret:
            self.frame = frame
        return ret, frame
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
