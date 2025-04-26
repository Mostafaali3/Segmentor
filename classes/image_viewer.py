import pyqtgraph as pg
from enums.viewerType import ViewerType
from PyQt5.QtWidgets import QFileDialog
import cv2


class ImageViewer(pg.ImageView):
    def __init__(self):
        super().__init__()
        self.getView().setBackgroundColor("#ed56f9")
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.getView().setAspectLocked(False)
        self.current_image = None
        self.viewer_type = None
        
    def update_plot(self):
        if self.current_image is not None:
            self.clear()
            view = self.getView()
            if self.viewer_type == ViewerType.INPUT:
                self.setImage(cv2.transpose(self.current_image.original_image))
            elif self.viewer_type == ViewerType.OUTPUT:
                self.setImage(cv2.transpose(self.current_image.modified_image))
            view.setLimits(xMin = 0, xMax=self.current_image.original_image.shape[1], yMin = 0, yMax = self.current_image.original_image.shape[0])