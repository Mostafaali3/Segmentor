from copy import deepcopy
import numpy as np
import cv2
from enums.type import Type


class Image():
    def __init__(self, data=None):
        self.__original_image = data
        self.__modified_image = deepcopy(data)
        self.__starter_function(data)

    def __starter_function(self, data):  # private function
        self.current_type = Type.NONE
        if data is not None:
            self.is_loaded = True
            if len(self.__original_image.shape) == 2:
                self.current_type = Type.GRAY
                self.__original_image = np.array(data, dtype=np.uint8)
            else:
                self.current_type = Type.RGB
                image_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                self.__original_image = image_rgb
            self.__modified_image = deepcopy(self.__original_image)

    def transfer_to_gray_scale(self):
        if len(self.__modified_image.shape) != 2:
            imported_image_gray_scale = np.dot(self.__modified_image[..., :3], [0.2989, 0.570, 0.1140])
            self.__modified_image = np.array(imported_image_gray_scale, dtype=np.uint8)
            
    def reset(self):
        self.__modified_image = deepcopy(self.__original_image)
        
    def handel_self_and_other_image_sizes(self, other):
        min_height = min(self.original_image.shape[0], other.original_image.shape[0])
        min_width = min(self.original_image.shape[1], other.original_image.shape[1])
        min_val = min(min_height, min_width)
        
        image_1 = cv2.resize(self.original_image, (min_val,min_val))
        image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        self.modified_image = image_1_gray
        
        image_2 = cv2.resize(other.original_image, (min_val, min_val))
        image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        other.modified_image = image_2_gray
        return image_1, image_2
        

    @property
    def original_image(self):
        return self.__original_image

    @property
    def modified_image(self):
        return self.__modified_image

    @modified_image.setter
    def modified_image(self, value):
        self.__modified_image = value
    @original_image.setter
    def original_image(self, value):
        self.__original_image = value