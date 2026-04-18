from PyQt5 import QtWidgets, QtCore
import sys
from PyQt5.QtGui import *
import cv2
import threading



class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle('oxxo.studio')

        self.setFixedSize(700, 600)
        self.ui()
        self.cap = cv2.VideoCapture(1)

    def ui(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(30, 20, 640,480))
        self.label.setStyleSheet("border: 5px solid black; border-radius: 10px;")
        self.label.setObjectName("label")

        self.textWidget = QtWidgets.QTextEdit(self)
        self.textWidget.setGeometry(QtCore.QRect(30, 520, 250, 40))
        self.textWidget.setStyleSheet("border-radius: 5px;background-color: #ECFAF8; color: black; font-size: 35px;")
        self.textWidget.setObjectName("textWidget")
        

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(300, 520, 100, 40))
        self.pushButton.setStyleSheet("border-radius: 5px;background-color: #ECFAF8; color: white; font-size: 16px;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.take_photo)
    
    def update_image(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.label.setPixmap(pixmap)
    def take_photo(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(f'{self.textWidget.toPlainText()}.jpg', frame)
            print(f"照片已保存為 {self.textWidget.toPlainText()}.jpg")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyWidget()
    video_thread = threading.Thread(target=MainWindow.update_image, daemon=True)
    video_thread.start()
    MainWindow.show()
    sys.exit(app.exec_())