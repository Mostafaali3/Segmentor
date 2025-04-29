import numpy as np


class Spectral_thresholding():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer


    def apply_spectral_thresholding(self, image):
        if self.output_image_viewer.current_image is not None:
            if len(image.shape) == 3:
                print("img is not gray")

        pdf = self.compute_histogram(image)
        overall_mean = 0.0

        for i in range(256):
            overall_mean += i * pdf[i]

        max_var = -np.inf
        best_low_thresh = 0
        best_high_thresh = 0

        for low_thresh in range(256):
            for high_thresh in range(low_thresh + 1, 256):
                left_weight, middle_weight, right_weight = 0.0, 0.0, 0.0
                left_sum, middle_sum, right_sum = 0.0, 0.0, 0.0
                left_mean, middle_mean, right_mean = 0.0, 0.0, 0.0

                for i in range(256):
                    if i <= low_thresh:
                        left_weight += pdf[i]
                        left_sum += i * pdf[i]
                    elif i <= high_thresh:
                        middle_weight += pdf[i]
                        middle_sum += i * pdf[i]
                    else:
                        right_weight += pdf[i]
                        right_sum += i * pdf[i]

                if left_weight == 0 or middle_weight == 0 or right_weight == 0: # iteration malhas lazma / divsion by zero
                    continue

                left_mean = left_sum / left_weight
                right_mean = right_sum / right_weight
                middle_mean = middle_sum / middle_weight

                var = (left_weight * (left_mean - overall_mean) ** 2 +
                            middle_weight * (middle_mean - overall_mean) ** 2 +
                            right_weight * (right_mean - overall_mean) ** 2)

                if var > max_var:
                    max_var = var
                    best_low_thresh = low_thresh
                    best_high_thresh = high_thresh

        return best_low_thresh, best_high_thresh



    def compute_histogram(self, image):
        # creates a 1d arr hist of size 256
        hist = np.zeros(256, dtype=np.float32)
        for pixel in image.flatten():
            hist[int(pixel)] += 1
        hist = hist / np.sum(hist)
        return hist







