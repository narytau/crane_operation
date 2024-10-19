import numpy as np
from scipy.ndimage import gaussian_filter1d

class MyGaussianFilter():
    def __init__(self, sigma, window_size):
        self.sigma = sigma
        self.window_size = window_size
        
    
    def process_frame(self, measurement):
        if self.data_window.shape[0] < self.window_size:
            self.data_window = np.vstack([self.data_window, measurement])
        else:
            self.data_window = np.roll(self.data_window, -1, axis=0)
            self.data_window[-1] = measurement

        smoothed_data = gaussian_filter1d(self.data_window, self.sigma, axis=0)
        
        self.current_smoothed_value = smoothed_data[-1]
        
    
    def set_data(self, data):
        self.data = data
        self.data_window = np.zeros((0, len(data)))
        
    def get_current_data(self):
        return self.current_smoothed_value