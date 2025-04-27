import cv2
import numpy as np


class Thresholder():
    def __init__(self,output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.threshold_type = None
        self.check_global_selection = False

    def apply_thresholding(self, threshold_type, thersh_val):
        # thesh_val passed 3la 7asb the method --> handle it later 
        if self.output_image_viewer.current_image is not None:
            if threshold_type == "LOCAL":
                self.restore_original_img()
                self.apply_local_thresholding(thersh_val)
            elif threshold_type == "GLOBAL" and self.check_global_selection:
                self.restore_original_img()
                self.apply_global_thresholding(thersh_val)

    def apply_global_thresholding(self, thersh_val):
        height, width = self.output_image_viewer.current_image.modified_image.shape
        thresholded_img = np.zeros((height, width), dtype=np.uint8)

        for row in range(height):
            for col in range(width):
                if self.output_image_viewer.current_image.modified_image[row, col] < thersh_val:
                    thresholded_img[row, col] = 0
                else:
                    thresholded_img[row, col] = 255
        print(f"thresh img {thresholded_img}")

        self.output_image_viewer.current_image.modified_image = thresholded_img


    def apply_local_thresholding(self, thersh_val):
        block_size = 11
        c= 2
        height, width = self.output_image_viewer.current_image.modified_image.shape
        thresholded_img = np.zeros((height, width), dtype=np.uint8)
        img = self.output_image_viewer.current_image.modified_image
        for i in range(height):
            for j in range(width):
                x_min = max(0, i - block_size // 2)
                y_min = max(0, j - block_size // 2)
                x_max = min(height -1, i + block_size // 2)
                y_max = min(width - 1, j + block_size // 2)
                block = img[x_min:x_max + 1, y_min:y_max + 1]
                # not calculated
                thresh = np.mean(block) - c

                if img[i, j] >= thresh:
                    thresholded_img[i, j] = 255

        self.output_image_viewer.current_image.modified_image= thresholded_img

    def restore_original_img(self):
        imported_image_gray_scale = cv2.cvtColor(self.output_image_viewer.current_image.original_image, cv2.COLOR_BGR2GRAY)
        self.output_image_viewer.current_image.modified_image = np.array(imported_image_gray_scale, dtype=np.uint8)