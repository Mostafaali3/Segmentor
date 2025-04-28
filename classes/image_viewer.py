import pyqtgraph as pg
from PyQt5 import Qt

from enums.viewerType import ViewerType
from PyQt5.QtWidgets import QFileDialog
import cv2


class ImageViewer(pg.ImageView):
    def __init__(self):
        super().__init__()
        self.getView().setBackgroundColor("#edf6f9")
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.getView().setAspectLocked(False)
        self.current_image = None
        self.viewer_type = None


    def mousePressEvent(self, event):


        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            print(f"Clicked at widget coordinates: ({x}, {y})")

            # Save clicked point for drawing
            self.clicked_point = (x, y)

            # Request repaint
            self.update()

            # Map to image coordinates if needed
            widget_width = self.width()
            widget_height = self.height()
            img_width = self.q_image.width()
            img_height = self.q_image.height()

            ratio_x = img_width / widget_width
            ratio_y = img_height / widget_height

            img_x = int(x * ratio_x)
            img_y = int(y * ratio_y)

            print(f"Clicked at image coordinates: ({img_y}, {img_x})")

            # You can call RegionGrowing here
            self.paintEvent()
            self.region_growing((img_y, img_x))

    def update_plot(self):
        if self.current_image is not None:
            self.clear()
            view = self.getView()
            if self.viewer_type == ViewerType.INPUT:
                self.setImage(cv2.transpose(self.current_image.original_image))
            elif self.viewer_type == ViewerType.OUTPUT:
                self.setImage(cv2.transpose(self.current_image.modified_image))
            view.setLimits(xMin = 0, xMax=self.current_image.original_image.shape[1], yMin = 0, yMax = self.current_image.original_image.shape[0])