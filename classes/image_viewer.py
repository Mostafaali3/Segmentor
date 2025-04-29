import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QBrush

from enums.viewerType import ViewerType
from PyQt5.QtWidgets import QFileDialog
import cv2
from enums.mode import Mode
from enums.segmentationType import SegmentationType


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
        self.current_mode = None
        self.current_segmentation_mode = None
        self.region_growing = None
        self.clicked_point = None
        self.plotted_point = None

    def mousePressEvent(self, event):
        if self.viewer_type == ViewerType.OUTPUT:
            if self.current_mode == Mode.SEGMENTATION:
                if self.current_segmentation_mode == SegmentationType.REGION_GROWING:
                    if event.button() == Qt.LeftButton:
                        x = event.pos().x()
                        y = event.pos().y()
                        print(f"Clicked at widget coordinates: ({x}, {y})")

                        # Save clicked point for drawing
                        self.clicked_point = (x, y)


                        # # Request repaint
                        # self.update()
                        #
                        # Map to image coordinates if needed
                        widget_width = self.width()
                        widget_height = self.height()
                        img_height, img_width, b = self.current_image.modified_image.shape

                        ratio_x = img_width / widget_width
                        ratio_y = img_height / widget_height

                        img_x = int(x * ratio_x)
                        img_y = int(y * ratio_y)

                        print(f"Clicked at image coordinates: ({img_y}, {img_x})")

                        self.region_growing.seed_x = img_x
                        self.region_growing.seed_y = img_y
                        
                        if self.plotted_point:
                            view = self.getView()
                            view.removeItem(self.plotted_point)
                        point = pg.ScatterPlotItem(x = [img_x], y=[img_y], brush='r', size=12)
                        self.getView().addItem(point)
                        self.plotted_point = point
                        
                        #
                        # # You can call RegionGrowing here
                        # self.update()
                        # self.region_growing((img_y, img_x))
                else:
                    if self.plotted_point:
                        self.getView().removeItem(self.plotted_point)
                        self.plotted_point = None
            else:
                if self.plotted_point:
                    self.getView().removeItem(self.plotted_point)
                    self.plotted_point = None
        else:
            if self.plotted_point:
                self.getView().removeItem(self.plotted_point)
                self.plotted_point = None
    # def paintEvent(self, event):
    #     painter = QPainter(self)
    #
    #     # Debug print to verify clicked_point
    #     print(f"Paint event called. Clicked point: {self.clicked_point}")
    #
    #     if self.clicked_point:
    #         x, y = self.clicked_point
    #         print(f"Drawing circle at: ({x}, {y})")
    #
    #         # Make circle very visible for testing
    #         painter.setPen(QPen(Qt.red, 5))  # Red outline
    #         painter.setBrush(QBrush(Qt.red))  # Red fill
    #         painter.drawEllipse(QPoint(x, y), 10, 10)  # Radius 10 circle


        
    def update_plot(self):
        if self.current_image is not None:
            self.clear()
            view = self.getView()
            if self.viewer_type == ViewerType.INPUT:
                self.setImage(cv2.transpose(self.current_image.original_image))
                view.setLimits(xMin = 0, xMax=self.current_image.original_image.shape[1], yMin = 0, yMax = self.current_image.original_image.shape[0])
            elif self.viewer_type == ViewerType.OUTPUT:
                self.setImage(cv2.transpose(self.current_image.modified_image))
                view.setLimits(xMin = 0, xMax=self.current_image.modified_image.shape[1], yMin = 0, yMax = self.current_image.modified_image.shape[0])
                if self.plotted_point:
                        self.getView().removeItem(self.plotted_point)
                        self.plotted_point = None