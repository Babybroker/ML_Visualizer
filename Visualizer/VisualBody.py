from matplotlib.pyplot import savefig
from datetime import datetime as dt

class VisualizerBody:
    def __init__(
            self,
            file_location
    ):
        self.file_location = file_location

    def save_func(self, save_visual, timestamp, folder_name, filename):
        save_timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        if save_visual:
            if timestamp is None:
                timestamp = save_timestamp
            filename = f'{self.file_location}/figs/model_{timestamp}.png' if folder_name is None \
                else folder_name + f'{filename}_{timestamp}.png'
            savefig(filename)
