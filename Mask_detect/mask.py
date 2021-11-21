#Ningning winter Giselle Karina
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model = load_model("model/mask_new2.h5")
class_names = ['Mask', 'NoMask']
net = cv2.dnn.readNetFromCaffe("model/deploy.prototxt", "model/model.caffemodel")
cap = cv2.VideoCapture("New video.mp4")
k, j = 0, 0
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame_ = frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            img = frame_[y1:y2, x1:x2]
            try:
                img_ = cv2.resize(img, (180, 180))
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_array = tf.keras.utils.img_to_array(img_)
                img_array = tf.expand_dims(img_array, 0)
                predictions = model.predict(img_array)
                score = round((100 * np.max(predictions)), 2)
                name = class_names[np.argmax(predictions)]
                if name == 'NoMask':
                    color_ = (0, 0, 255)
                else:
                    color_ = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_, 2)
                cv2.putText(frame, str(score), (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color_, 2)
            except:
                print("none Face")
    f_ = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow("frame", f_)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
