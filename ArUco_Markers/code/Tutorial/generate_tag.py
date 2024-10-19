import numpy as np
import matplotlib.pyplot as plt

from cv2 import aruco

# aruco辞書の生成
dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# determine ID (適当な整数)
marker_id = 10

# marker size
size_mark = 100

# imgの作成
image = aruco.generateImageMarker(dict_aruco, marker_id, size_mark)

plt.imshow(image, cmap='gray')
plt.show()