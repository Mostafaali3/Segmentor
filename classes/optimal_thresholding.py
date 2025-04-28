import numpy as np
from classes.thresholder import Thresholder


class Optimal_thresholding():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.epsilon = 0.002
        self.thresholder = Thresholder(self.output_image_viewer)


    def apply_optimal_thresholding(self, threshold_type):
        if self.output_image_viewer.current_image is not None:
            # conversion to gray
            if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
                self.output_image_viewer.current_image.transfer_to_gray_scale()

            # starting with initial threshold
            current_thresh = self.compute_initial_threshold()
            pdf = self.thresholder.compute_histogram()

            while True:
                # splits pixels into two groups (object and background) based on the current threshold
                # calculates  means of each group and updates the threshold as the avg of these means
                mean1, mean2 = 0,0

                new_thresh = (mean1 + mean2) / 2
                if abs(current_thresh - new_thresh) < self.epsilon:
                    self.thresholder.apply_thresholding(threshold_type, current_thresh)
                    break
                current_thresh = new_thresh


    def compute_initial_threshold(self):
        height, width = self.output_image_viewer.current_image.modified_image.shape
        # corner pixel intensities
        corner_pixels = [
            self.output_image_viewer.current_image.modified_image[0,0],
            self.output_image_viewer.current_image.modified_image[0, width -1],
            self.output_image_viewer.current_image.modified_image[height -1, 0],
            self.output_image_viewer.current_image.modified_image[height-1, width-1]
        ]

        init_thresh = np.mean(corner_pixels)
        return init_thresh





