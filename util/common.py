import cv2
import tkinter as tk
from tkinter import filedialog

def open_dialog():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()

def cv_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img)
    cv2.waitKey(0)


class TimeCost:
    tm = 0

    def __init__(self):
        self.tm = cv2.TickMeter()
        self.tm.start()

    def getCostTime(self):
        self.tm.stop()
        return self.tm.getTimeSec()
