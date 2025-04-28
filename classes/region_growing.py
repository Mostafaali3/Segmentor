import collections

import cv2
import numpy
import numpy as np
from PyQt5 import Qt
from PyQt5.QtGui import QPainter, QPen


class region_growing():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer

    def paintEvent(self):
        painter = QPainter(self)

        # If a point is selected, draw a small circle
        if self.clicked_point:
            painter.setPen(QPen(Qt.red, 5))  # Red color, 5px width
            x, y = self.clicked_point
            painter.drawEllipse(x - 3, y - 3, 6, 6)  # Small circle centered at (x,y)

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

    def apply_region_growing(self, seed, threshold):
        if self.output_image_viewer.current_image is not None:
            height, width = self.output_image_viewer.current_image.modified_image.shape
            mask = np.zeros((height, width), np.uint8)

            if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
                print("img is not gray")
                self.output_image_viewer.current_image.transfer_to_gray_scale()

            seed_x, seed_y = seed
            base_seed = self.output_image_viewer.current_image[seed_x, seed_y]

            queue = collections.deque()
            queue.append((seed_x, seed_y))

            while queue:
                seed_x, seed_y = queue.pop(0)
                if seed_x <0 or seed_x > width or seed_y < 0 or seed_y > height:
                    continue

                if mask[seed_x, seed_y]:
                    continue

                if(abs(self.output_image_viewer.current_image[seed_x, seed_y] - base_seed)).all() <= threshold:
                    mask[seed_x, seed_y] = 255

                    queue.append((seed_x - 1, seed_y))
                    queue.append((seed_x, seed_y - 1))
                    queue.append((seed_x + 1, seed_y))
                    queue.append((seed_x, seed_y - 1))

