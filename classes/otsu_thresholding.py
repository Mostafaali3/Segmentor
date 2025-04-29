import numpy as np


class Otsu_thresholding():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer

    def apply_otsu_thresholding(self, image):
        if self.output_image_viewer.current_image is not None:
            if len(image.shape) == 3:
                print("img is not gray")
                return None

        # max_intensity = np.max(image)
        # thresh_range = range(max_intensity + 1) # could ve been 256 as we r sure that the img is gray
        thresh_range = range(256)

        min_criteria = np.inf
        best_thresh = 0

        pdf = self.compute_histogram(image)

        for thresh in thresh_range:
            criteria = self.get_best_threshold(pdf, thresh)
            if criteria < min_criteria:
                min_criteria = criteria
                best_thresh = thresh

        return best_thresh

    def get_best_threshold(self, pdf, thresh):
        background_weight, foreground_weight = 0.0, 0.0
        background_sum, foreground_sum = 0.0, 0.0
        background_var, foreground_var = 0.0, 0.0

        for i in range(256):
            if i <= thresh:
                background_weight += pdf[i]
                background_sum += i * pdf[i]
            else:
                foreground_weight += pdf[i]
                foreground_sum += i * pdf[i]

        background_mean = background_sum / background_weight if background_weight > 0 else 0
        foreground_mean = foreground_sum / foreground_weight if foreground_weight > 0 else 0



        for i in range(256):
            if i <= thresh:
                background_var += ((i - background_mean) ** 2) * pdf[i]
            else:
                foreground_var += ((i - foreground_mean) ** 2) * pdf[i]

        background_var = background_var / max(background_weight, 1e-10)
        foreground_var = foreground_var / max(foreground_weight, 1e-10)

        return background_weight * background_var + foreground_weight * foreground_var



    def compute_histogram(self, image):
        # creates a 1d arr hist of size 256
        hist = np.zeros(256, dtype=np.float32)
        for pixel in image.flatten():
            hist[int(pixel)] += 1
        hist = hist / np.sum(hist)
        return hist





