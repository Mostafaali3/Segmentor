import collections

import cv2
import numpy
import numpy as np


class region_growing():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
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

