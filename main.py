
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QVBoxLayout, QSlider, QComboBox, QPushButton, \
    QStackedWidget, QWidget, QFileDialog, QRadioButton, QDialog, QLineEdit, QHBoxLayout, QSpinBox
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
from classes.image import Image
from classes.image_viewer import ImageViewer
from classes.region_growing import region_growing
from enums.viewerType import ViewerType
from classes.controller import Controller
import cv2
from classes.Agglomerative import Agglomerative

from classes.Kmeans import Kmeans

from classes.meanShiftSegmenter import MeanShiftSegmenter
from classes.thresholder import Thresholder
from enums.mode import Mode
from enums.segmentationType import SegmentationType


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
        
        self.input_viewer.current_mode = Mode.SEGMENTATION
        self.output_viewer.current_mode = Mode.SEGMENTATION
        
        self.threshold_button = self.findChild(QPushButton, "thresholding_button")
        self.threshold_button.clicked.connect(self.on_threshold_mode_button_pressed)
        self.segmentation_button = self.findChild(QPushButton, "segmentation_button")
        self.segmentation_button.clicked.connect(self.on_segmentation_mode_button_pressed)
        self.modes_stacked_widget = self.findChild(QStackedWidget, "stackedWidget")

        self.kmeans = Kmeans(self.input_viewer, self.output_viewer)
        self.agglomerative = Agglomerative(self.input_viewer, self.output_viewer)
        
        self.controller = Controller(self.input_viewer, self.output_viewer)
        
        self.browse_button = self.findChild(QPushButton, "browse")
        self.browse_button.clicked.connect(self.browse_)

        self.region_growing = region_growing(self.output_viewer)
        self.output_viewer.region_growing = self.region_growing


        
        self.segmentation_combobox = self.findChild(QComboBox, "segmentation_type")
        self.segmentation_combobox.currentIndexChanged.connect(self.on_segmentation_combobox_index_changed)
        
        self.apply_segmentation_button = self.findChild(QPushButton, "apply_segmentation_button")
        self.apply_segmentation_button.clicked.connect(self.on_apply_button_clicked)
        
        self.mean_shift_segmenter = MeanShiftSegmenter(self.input_viewer, self.output_viewer)
        
        self.segmentation_browse_button = self.findChild(QPushButton, "browse_segmentation")
        self.segmentation_browse_button.clicked.connect(self.browse_)

        self.segmentation_slider_counter_label = self.findChild(QLabel, "iterations_label")
        
        self.segmentation_slider = self.findChild(QSlider, "iterations_slider")
        self.segmentation_slider.valueChanged.connect(self.on_segmentation_slider_value_changed)
        
        self.block_size_slider = self.findChild(QSlider, "threshold_value_slider")
        self.block_size_slider.valueChanged.connect(self.on_block_size_slider_value_changed)
        
        self.block_size_slider_label = self.findChild(QLabel, "therolding_value_label")
        
        self.reset_button = self.findChild(QPushButton, "pushButton_2")
        self.reset_button.clicked.connect(self.reset)


        # tresholding stuff (7ad yefkarny amsa7 el comment da)
        self.apply_thresholding_button = self.findChild(QPushButton, "apply_threshold")
        self.apply_thresholding_button.clicked.connect(self.apply_thresholding)
        self.thresholder = Thresholder(self.output_viewer)
        self.threshold_type = None

        self.local_threshold = self.findChild(QRadioButton, "local_thresholding")
        self.global_threshold = self.findChild(QRadioButton, "global_thresholding")

        self.local_threshold.toggled.connect(self.on_threshold_selected)
        self.global_threshold.toggled.connect(self.on_threshold_selected)

        self.thresholding_method_selector = self.findChild(QComboBox, 'threshold_type')
        self.slider_label = self.findChild(QLabel, "label_8")
        self.slider_label.setText("no. of clusters:")

        
        
    def on_threshold_mode_button_pressed(self):
        self.modes_stacked_widget.setCurrentIndex(0)
        self.input_viewer.current_mode = Mode.THRESHOLDING
        self.output_viewer.current_mode = Mode.THRESHOLDING
        
    def on_segmentation_mode_button_pressed(self):
        self.modes_stacked_widget.setCurrentIndex(1)
        self.input_viewer.current_mode = Mode.SEGMENTATION
        self.output_viewer.current_mode = Mode.SEGMENTATION
        
        
    def on_segmentation_slider_value_changed(self):
        self.segmentation_slider_counter_label.setText(f"{self.segmentation_slider.value()}")
        
    def on_segmentation_combobox_index_changed(self):
        '''
        this function is for any changes in the inputs labels 
        '''
        text = self.segmentation_combobox.currentText()
        slider_label = self.findChild(QLabel, "label_8")
        if text == 'K-means':
            slider_label.setText("no. of clusters:")
            self.output_viewer.current_segmentation_mode = SegmentationType.KNN
            self.segmentation_slider.setRange(1, 10)
        elif text == 'Mean shifting':
            slider_label.setText("kernel size")
            self.segmentation_slider.setMaximum(300)
            self.output_viewer.current_segmentation_mode = SegmentationType.MEAN_SHIFT
            pass #write your code her
        
        elif text == 'Region growing':
            slider_label.setText("threshold:")
            self.output_viewer.current_segmentation_mode = SegmentationType.REGION_GROWING
            self.segmentation_slider.setRange(1, 250)
            self.segmentation_slider.setValue(100)
            pass #write your code her
        
        else:
            slider_label.setText("no. of clusters:")
            self.output_viewer.current_segmentation_mode = SegmentationType.AGGLOMERATIVE
            self.segmentation_slider.setRange(1, 10)
            pass #write your code her
        
    def on_apply_button_clicked(self):
        text = self.segmentation_combobox.currentText()
        if text == 'K-means':
            self.kmeans.apply_kmeans(self.segmentation_slider.value())


        elif text == 'Mean shifting':
            self.mean_shift_segmenter.apply_mean_shift(self.segmentation_slider.value())
        
        elif text == 'Region growing':
            self.region_growing.apply_region_growing(self.segmentation_slider.value())
        else:
            self.agglomerative.apply_agglomerative(self.segmentation_slider.value())
        self.controller.update()


    def apply_thresholding(self):
        thresholding_method = self.thresholding_method_selector.currentText()
        print(f"thresh type {self.threshold_type}")
        print(f"thresh method {thresholding_method}")
        print(f"block size value {int(self.block_size_slider.value())}")
        block_size = int(self.block_size_slider.value())

        self.thresholder.apply_thresholding(self.threshold_type, thresholding_method, block_size)
        self.controller.update()

    def on_threshold_selected(self):
        # never calling it twice
        if self.sender().isChecked():
            if self.sender() == self.local_threshold:
                self.threshold_type = "LOCAL"
                self.thresholder.check_global_selection = False
            elif self.sender() == self.global_threshold:
                self.threshold_type = "GLOBAL"
                self.thresholder.check_global_selection = True
    
    def on_block_size_slider_value_changed(self):
        self.block_size_slider_label.setText(f"{self.block_size_slider.value()}")

        
        
        
        

        
    def browse_(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpeg *.jpg *.png *.JPG);;All Files (*)')
        if file_path:
            if file_path.endswith('.jpeg') or file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.JPG'):
                temp_image = cv2.imread(file_path)
                image = Image(temp_image)
                self.input_viewer.current_image = image
                self.output_viewer.current_image = image
                height, width, z = self.output_viewer.current_image.modified_image.shape
                
                self.block_size_slider.setRange(3, max(height, width) )
                self.controller.update()
        
    def reset(self):
        self.output_viewer.current_image.reset()
        self.controller.update()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())