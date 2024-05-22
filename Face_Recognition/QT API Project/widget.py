# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import cv2
import numpy as np

from PySide6 import QtWidgets, QtUiTools, QtGui, QtCore

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

import pickle

from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder



class Widget(QWidget):
    def __init__(self):
        super(Widget, self).__init__()
        self.ui = self.load_ui()

        # Убираем рамки и разрешаем перетаскивание объектов

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.center()



        self.capture_state = False

        self.capture = None

        # Buttons
        self.ui.startButton.clicked.connect(self.start_button_clicked)
        self.ui.stopButton.clicked.connect(self.stop_button_clicked)
        self.ui.roll_up_button.clicked.connect(lambda: self.showMinimized())
        self.ui.close_button.clicked.connect(lambda: self.close())



        self.face_detection_model = load_model('face_detection.h5')
        self.feature_extraction_model = FaceNet()


        self.face_recognition_model = pickle.load(open('finalmodel_v2.sav', 'rb'))

        self.names = ['Nick_Korol', 'Anna_Osipchik', 'Ilya_Lisov']



        npz_file = np.load('faces_v3.npz')
        self.y = npz_file['arr_1']

        self.encoder = LabelEncoder()
        self.encoder.fit(self.y)
        self.y = self.encoder.transform(self.y)
        # self.siamese_model = load_model('siamese_model.h5',
        #                          custom_objects={'L1Dist':L1Dist,
        #                                          'BinatyCrossentropy':tf.losses.BinaryCrossentropy})
        # self.photo_to_compare =  tf.data.Dataset.list_files(test_path+'\*.jpg').take(3)
        # self.photo_to_compare = self.preprocess('data\\b9f24baa-ddfb-11ee-8abb-185e0f1d68b5.jpg')
        # self.photo_to_compare = self.photo_to_compare.map(self.preprocess)

    # def preprocess(self, file_path):
    #     byte_img = tf.io.read_file(file_path)
    #     img = tf.io.decode_jpeg(byte_img)
    #     img = tf.image.resize(img, (105, 105))
    #     img = img / 255.
    #     return img

    def get_embedding(self, face_img):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)

        yhat = self.feature_extraction_model.embeddings(face_img)
        return yhat[0]

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        ui = loader.load(ui_file, self)
        ui_file.close()
        return ui

    def start_button_clicked(self):
        if self.capture_state == False:
            self.capture_state = True
            self.capture = cv2.VideoCapture(1)

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

    def stop_button_clicked(self):
        self.capture_state = False
        if self.capture and type(self.capture) == cv2.VideoCapture and self.capture.isOpened():
            self.capture.release()
        self.capture = None
        self.ui.label.setText('Waiting for starting capturing...')
            
            


    def update_frame(self):
        if self.capture_state == True:
            ret, frame = self.capture.read()
            frame = self.face_detection(frame)

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_img)
            self.ui.label.setPixmap(pixmap)

    def face_detection(self, frame):
        frame = frame[50:500, 50:500,:]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # transformation for detection
        resized_for_detection = tf.image.resize(frame_rgb, (120,120))


        yhat = self.face_detection_model.predict(np.expand_dims(resized_for_detection/255,0))

        sample_coords = yhat[1][0]
        sample_coords = np.multiply(sample_coords, [450, 450, 450, 450]).astype(int)



        if yhat[0] > 0.5:

            # recognition_result = self.si_amesemodel.predict([self.photo_to_compare, resized_for_recognition])
            #
            #
            #
            # print(recognition_result[0][0])



            face_coord = frame[sample_coords[1]:sample_coords[3], sample_coords[0]:sample_coords[2]]
            face_coord = cv2.resize(face_coord, (160, 160))

            embedding = self.get_embedding(face_coord)
            ypred = self.face_recognition_model.predict([embedding])


            color = (0, 255, 0) if ypred[0] in self.names else (0, 0, 255)
        # Controls the main rectangle
            cv2.rectangle(frame,
                        sample_coords[:2],
                        sample_coords[2:],
                                color, 2)
            # Controls the label rectangle
            cv2.rectangle(frame,
                        tuple(np.add(sample_coords[:2],
                                        [0,-30])),
                        tuple(np.add(sample_coords[:2],
                                        [80,0])),
                                color, -1)
            #
            # # Controls the text rendered
            cv2.putText(frame, ypred[0], tuple(np.add(sample_coords[:2],
                                                [0,-5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return frame

    def closeEvent(self, event):
        if self.capture and type(self.capture) == cv2.VideoCapture and self.capture.isOpened():
            self.capture.release()
        event.accept()

    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == "__main__":
    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec_())
