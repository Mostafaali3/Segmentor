import numpy as np


class Optimal_thresholding():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.epsilon = 0.002

    def apply_optimal_thresholding(self, threshold_type):
        if self.output_image_viewer.current_image is not None:
            # conversion to gray
            if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
                self.output_image_viewer.current_image.transfer_to_gray_scale()

            # starting with initial threshold
            current_thresh = self.compute_initial_threshold()
            pdf = self.compute_histogram()

            # breaks when convergence is achieved
            while True:
                # splits pixels into two groups (object and background) based on the current threshold
                # calculates  means of each group and updates the threshold as the avg of these means
                background_sum, object_sum = 0.0, 0.0
                background_cdf, object_cdf = 0.0, 0.0

                # looping over all intensities
                for i in range(256):
                    #object
                    if i <= current_thresh:
                        object_sum += i * pdf[i]
                        object_cdf += pdf[i]
                    # background
                    else:
                        background_sum += i* pdf[i]
                        object_cdf += pdf[i]

                # mean = weighted sum / cumulative probability
                background_mean = background_sum / max(background_cdf, 1e-10) if background_cdf > 0 else 0
                object_mean = object_sum / max(object_cdf, 1e-10) if object_cdf > 0 else 0

                new_thresh = (background_mean + object_mean) / 2
                if abs(current_thresh - new_thresh) < self.epsilon:
                    # self.thresholder.apply_thresholding(threshold_type, current_thresh)
                    return current_thresh
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

    def compute_histogram(self):
        # creates a 1d arr hist of size 256
        hist = np.zeros(256, dtype=np.float32)
        for pixel in self.output_image_viewer.current_image.modified_image.flatten():
            hist[int(pixel)] += 1
        hist = hist / np.sum(hist)
        return hist





