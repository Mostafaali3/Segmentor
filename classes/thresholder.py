import cv2
import numpy as np
from classes.optimal_thresholding import Optimal_thresholding


class Thresholder():
    def __init__(self,output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.threshold_type = None
        self.check_global_selection = False
        self.optimal_thresholding = Optimal_thresholding(output_image_viewer)

    def apply_thresholding(self, threshold_type, thresh_method):
        # thesh_val passed 3la 7asb the method --> handle it later
        if self.output_image_viewer.current_image is not None:
            if threshold_type == "LOCAL":
                self.restore_original_img()
                self.apply_local_thresholding(thresh_method)
            elif threshold_type == "GLOBAL" and self.check_global_selection:
                self.restore_original_img()
                self.apply_global_thresholding(thresh_method)

    def apply_global_thresholding(self, thresh_method):
        height, width = self.output_image_viewer.current_image.modified_image.shape
        thresholded_img = np.zeros((height, width), dtype=np.uint8)

        if thresh_method == "Optimal thresholding":
            thersh_val = self.apply_optimal_thresholding(self.output_image_viewer.current_image.modified_image)
        elif thresh_method == "Otsu thresholding":
            pass
        elif thresh_method == "Spectral thresholding":
            pass

        for row in range(height):
            for col in range(width):
                if self.output_image_viewer.current_image.modified_image[row, col] < thersh_val:
                    thresholded_img[row, col] = 0
                else:
                    thresholded_img[row, col] = 255
        print(f"thresh img {thresholded_img}")

        self.output_image_viewer.current_image.modified_image = thresholded_img

    def apply_local_thresholding(self, thresh_method):
        height, width = self.output_image_viewer.current_image.modified_image.shape
        thresholded_img = np.zeros((height, width), dtype=np.uint8)
        block_size = 32
        image = self.output_image_viewer.current_image.modified_image

        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                block_end_row = min(row + block_size, height)
                block_end_col = min(col + block_size, width)
                block = image[row:block_end_row, col:block_end_col]

                if thresh_method == "Optimal thresholding":
                    thresh_val = self.apply_optimal_thresholding(block)
                elif thresh_method == "Otsu thresholding":
                    thresh_val = 128
                elif thresh_method == "Spectral thresholding":
                    thresh_val = 128
                else:
                    thresh_val = 128

                block_thresholded = np.where(block < thresh_val, 0, 255)
                thresholded_img[row:block_end_row, col:block_end_col] = block_thresholded

        self.output_image_viewer.current_image.modified_image = thresholded_img


    def apply_optimal_thresholding(self, image):
        optimal_thresh = self.optimal_thresholding.apply_optimal_thresholding(image)
        return optimal_thresh

    def apply_otsu_thresholding(self):
        pass


    def apply_specular_thresholding(self):
        pass

    def restore_original_img(self):
        imported_image_gray_scale = cv2.cvtColor(self.output_image_viewer.current_image.original_image, cv2.COLOR_BGR2GRAY)
        self.output_image_viewer.current_image.modified_image = np.array(imported_image_gray_scale, dtype=np.uint8)
        print("img is now gray")

    # def compute_histogram(self):
    #     # creates a 1d arr hist of size 256
    #     hist = np.zeros(256, dtype=np.float32)
    #     for pixel in self.output_image_viewer.current_image.modified_image.flatten():
    #         hist[int(pixel)] += 1
    #     hist = hist / np.sum(hist)
    #     return hist
