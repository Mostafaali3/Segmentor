
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QVBoxLayout, QSlider, QComboBox, QPushButton, \
    QStackedWidget, QWidget, QFileDialog, QRadioButton, QDialog, QLineEdit, QHBoxLayout, QSpinBox
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
from classes.image import Image
from classes.image_viewer import ImageViewer
from enums.viewerType import ViewerType
from classes.controller import Controller
import cv2
from classes.meanShiftSegmenter import MeanShiftSegmenter

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        
        self.input_viewer_layout = self.findChild(QVBoxLayout,'input_layout')
        self.input_viewer = ImageViewer()
        self.input_viewer.viewer_type = ViewerType.INPUT
        self.input_viewer_layout.addWidget(self.input_viewer)
        
        self.output_viewer_layout = self.findChild(QVBoxLayout,'output_layout')
        self.output_viewer = ImageViewer()
        self.output_viewer.viewer_type = ViewerType.OUTPUT
        self.output_viewer_layout.addWidget(self.output_viewer)
        
        self.threshold_button = self.findChild(QPushButton, "thresholding_button")
        self.threshold_button.clicked.connect(self.on_threshold_mode_button_pressed)
        self.segmentation_button = self.findChild(QPushButton, "segmentation_button")
        self.segmentation_button.clicked.connect(self.on_segmentation_mode_button_pressed)
        self.modes_stacked_widget = self.findChild(QStackedWidget, "stackedWidget")
        
        self.controller = Controller(self.input_viewer, self.output_viewer)
        
        self.browse_button = self.findChild(QPushButton, "browse")
        self.browse_button.clicked.connect(self.browse_)
        
        self.segmentation_combobox = self.findChild(QComboBox, "segmentationtype")
        self.segmentation_combobox.currentIndexChanged.connect(self.on_segmentation_combobox_index_changed)
        
        self.apply_segmentation_button = self.findChild(QPushButton, "apply_segmentation_button")
        self.apply_segmentation_button.clicked.connect(self.on_apply_button_clicked)
        
        self.mean_shift_segmenter = MeanShiftSegmenter(self.input_viewer, self.output_viewer)
        
        self.segmentation_browse_button = self.findChild(QPushButton, "browse_segmentation")
        self.segmentation_browse_button.clicked.connect(self.browse_)
        
        
    def on_threshold_mode_button_pressed(self):
        self.modes_stacked_widget.setCurrentIndex(0)
    def on_segmentation_mode_button_pressed(self):
        self.modes_stacked_widget.setCurrentIndex(1)
        
    def on_segmentation_combobox_index_changed(self):
        '''
        this function is for any changes in the inputs labels 
        '''
        
        text = self.segmentation_combobox.currentText()
        slider_label = self.findChild(QLabel, "label_8")
        if text == 'K-nearest neighbor (KNN)':
            slider_label.setText("Max no. of iterations:")

        elif text == 'Mean shifting':
            slider_label.setText("kernel size")
            pass #write your code here
        
        elif text == 'Region growing':
            slider_label.setText("Max no. of iterations:")
            pass #write your code here
        
        else:
            slider_label.setText("Max no. of iterations:")
            pass #write your code here
        
    def on_apply_button_clicked(self):
        text = self.segmentation_combobox.currentText()
        if text == 'K-nearest neighbor (KNN)':
            pass
        elif text == 'Mean shifting':
            self.mean_shift_segmenter.apply_mean_shift(50)
            pass #write your code here
        
        elif text == 'Region growing':
            pass #write your code here
        else:
            pass #write your code here
        self.controller.update()
        
    def browse_(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpeg *.jpg *.png *.JPG);;All Files (*)')
        if file_path:
            if file_path.endswith('.jpeg') or file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.JPG'):
                temp_image = cv2.imread(file_path)
                image = Image(temp_image)
                self.input_viewer.current_image = image
                self.output_viewer.current_image = image
                self.controller.update()
        
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())