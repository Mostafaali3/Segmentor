import numpy as np


class Optimal_thresholding():
    def __init__(self, output_image_viewer):
        self.output_image_viewer = output_image_viewer
        self.epsilon = 0.002

    def apply_optimal_thresholding(self, image):
        cnt = 0
        if self.output_image_viewer.current_image is not None:
            if len(image.shape) == 3:
                print("img is not gray")
            # # conversion to gray
            # if len(self.output_image_viewer.current_image.modified_image.shape) == 3:
            #     self.output_image_viewer.current_image.transfer_to_gray_scale()

            # starting with initial threshold
            current_thresh = self.compute_initial_threshold(image)
            pdf = self.compute_histogram(image)

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
                        background_cdf += pdf[i]

                # mean = weighted sum / cumulative probability
                background_mean = background_sum / background_cdf if background_cdf > 0 else 0
                object_mean = object_sum / object_cdf if object_cdf > 0 else 0

                print(f"background_mean is {background_mean}")
                print(f"object_mean is {object_mean}")

                new_thresh = (background_mean + object_mean) / 2
                print(f"new_thresh is {new_thresh}")
                print(f"current_thresh is {current_thresh}")
                cnt+= 1
                if abs(current_thresh - new_thresh) < self.epsilon:
                    # self.thresholder.apply_thresholding(threshold_type, current_thresh)
                    print(f"cnt is {cnt}")
                    return current_thresh
                current_thresh = new_thresh


    def compute_initial_threshold(self, image):
        height, width = image.shape
        # corner pixel intensities
        corner_pixels = [
            image[0,0],
            image[0, width -1],
            image[height -1, 0],
            image[height-1, width-1]
        ]

        init_thresh = np.mean(corner_pixels)
        return init_thresh

    def compute_histogram(self, image):
        # creates a 1d arr hist of size 256
        hist = np.zeros(256, dtype=np.float32)
        for pixel in image.flatten():
            hist[int(pixel)] += 1
        hist = hist / np.sum(hist)
        return hist







