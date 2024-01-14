from yoloface import face_analysis
from IPython.display import display
from PIL import Image
import numpy
import os
import cv2

                            #  only first time.
                            #  Automatically  create folder .yoloface on cwd.
# def face_detect(img_input):
#     img,box,conf=face.face_detection(img_input, model='tiny')
#     print(box)                  # box[i]=[x,y,w,h]
#     print(conf)                 #  value between(0 - 1)  or probability
#     face.show_output(img,box)

def save_detected_face(img):
    face_detector=face_analysis()        #  Auto Download a large weight files from Google Drive.

    # 检测人脸并获取人脸边界框
    _, face_boxes, _ = face_detector.face_detection(img, model='tiny')

    # 提取被框出的区域并保存为 JPG 图像
    for i, box in enumerate(face_boxes):
        x, y, w, h = box
        print(x,y,w,h)
        cropped_face = img[y:y+w, x:x+h]  # 提取被框出的区域
        
