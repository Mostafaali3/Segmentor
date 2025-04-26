class Controller():
    def __init__(self, input_viewer, output_viewer):
        self.input_viewer = input_viewer
        self.output_viewer = output_viewer
        
    def update(self):
        self.input_viewer.update_plot()
        self.output_viewer.update_plot()
