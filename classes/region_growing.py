import collections

import cv2
import numpy
import numpy as np
from PyQt5 import Qt
from PyQt5.QtGui import QPainter, QPen


class region_growing():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.seed_x = None
        self.seed_y = None



    def apply_region_growing(self, threshold = 100):
        if self.output_image_viewer.current_image is not None:
            if self.seed_x is not None and self.seed_y is not None:
                height, width, b = self.output_image_viewer.current_image.modified_image.shape
                mask = np.zeros((height, width), np.uint8)

                # if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
                #     print("img is not gray")
                #     self.output_image_viewer.current_image.transfer_to_gray_scale()

                # base_seed = self.output_image_viewer.current_image.modified_image[self.seed_y, self.seed_x]

                queue = collections.deque()
                queue.append((self.seed_y, self.seed_x))

                while queue:
                    y, x = queue.popleft()

                    seed_value = self.output_image_viewer.current_image.modified_image[self.seed_y, self.seed_x]
                    if x < 0 or x >= width or y < 0 or y >= height:
                        continue

                    if mask[y, x]:
                        continue

                    current_value = self.output_image_viewer.current_image.modified_image[y, x]
                    # For color images, calculate distance in color space
                    color_distance = np.sqrt(np.sum((current_value.astype(np.float32) -
                                                     seed_value.astype(np.float32)) ** 2))
                    if color_distance <= threshold:
                        mask[y, x] = 255

                        queue.append((y - 1, x))
                        queue.append((y, x - 1))
                        queue.append((y + 1, x))
                        queue.append((y, x + 1))

                self.output_image_viewer.current_image.modified_image = mask
