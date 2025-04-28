from classes.image_viewer import ImageViewer
import numpy as np
import cv2


class MeanShiftSegmenter():
    def __init__(self, input_viewer:ImageViewer, output_viewer:ImageViewer):
        self.input_viewer = input_viewer
        self.output_viewer = output_viewer
        
    def apply_mean_shift(self, kernel_size=30, max_iters = 300, stop_thresh=1e-3):
        image = self.input_viewer.current_image.modified_image
        image = cv2.resize(image, (64,64))
        features = []
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                features.append(np.array([image[y][x][0], image[y][x][1], image[y][x][2], x, y]))
        features = np.array(features)
        feature_points =np.copy(features)
        shifted_points = np.zeros_like(feature_points)
        
        for i, point in enumerate(feature_points):
            mean = point
            for _ in range(max_iters):
                distances = np.linalg.norm(features-mean)
                within_distance = features[distances < kernel_size]    
                
                if len(within_distance) == 0:
                    break
                
                new_mean = np.mean(within_distance)
                
                shift = np.linalg.norm(new_mean, mean)
                if shift < stop_thresh:
                    break
                
                mean = new_mean
            shifted_points[i] = mean
        
        