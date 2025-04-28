from classes.image_viewer import ImageViewer

class MeanShiftSegmenter():
    def __init__(self, input_viewer, output_viewer):
        self.input_viewer = input_viewer
        self.output_viewer = output_viewer
        
    def apply_mean_shift(self, kernel_size):
        image = self.input_viewer.current_image.modified_image
        print(image[0])