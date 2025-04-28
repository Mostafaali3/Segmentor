from classes.image_viewer import ImageViewer
import numpy as np
import cv2
from copy import copy

class MeanShiftSegmenter():
    def __init__(self, input_viewer:ImageViewer, output_viewer:ImageViewer):
        self.input_viewer = input_viewer
        self.output_viewer = output_viewer
        
    def apply_mean_shift(self, kernel_size=40, max_iters = 300, stop_thresh=1e-3):
        image = copy(self.input_viewer.current_image.modified_image)
        image = cv2.resize(image, (100,100))
        features = []
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                features.append(np.array([image[y][x][0], image[y][x][1], image[y][x][2], x, y]))
        features = np.array(features)
        feature_points =np.copy(features)
        shifted_points = np.zeros_like(feature_points)
        print(feature_points[0])
        for i, point in enumerate(feature_points):
            mean = point
            for _ in range(max_iters):
                distances = np.linalg.norm(features-mean, axis=1)
                within_distance = features[distances < kernel_size]    
                within_distances = distances[distances < kernel_size]
                if len(within_distance) == 0:
                    break
                weights = np.exp(-(within_distances**2) / (2 * (kernel_size**2)))
                new_mean = np.average(within_distance, axis=0, weights=weights)
                
                shift = np.linalg.norm(new_mean- mean)
                if shift < stop_thresh:
                    break
                
                mean = new_mean
            shifted_points[i] = mean
            
            
        image = shifted_points[:, :3].reshape(image.shape[0], image.shape[1], 3)
        self.input_viewer.current_image.modified_image = image
        
        print(shifted_points[0])