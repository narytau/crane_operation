import numpy as np

def sliding_time_window(array, window_size, step_size):
    """
    Args:
    array (np.ndarray): original array (N x features)
    window_size (int): --> feature_num
    step_size (int): step size to move window

    Returns:
    List[np.ndarray]
    """

    data_num, feature_num = array.shape[0], array.shape[1]
    data_window = np.zeros((1 + int((data_num - window_size) / step_size), window_size, feature_num))
    start_idx = 0
    id = 0

    if step_size >= 1:
        while start_idx + window_size <= array.shape[0]:
            window = array[start_idx:start_idx + window_size]
            data_window[id, :, :] = window
            start_idx += step_size
            id += 1
    else:
        print("Change step size")
    
    return data_window
