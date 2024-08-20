import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# ガウスカーネルの標準偏差
sigma = 5
window_size = 100  # ウィンドウサイズを調整

# data_window を NumPy 配列として初期化
data_window = np.zeros((0, 2))

def apply_gaussian_smoothing(data, sigma):
    # ガウスカーネルをデータに適用
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=0)
    return smoothed_data

def process_frame(measurement):
    global data_window

    # ウィンドウに新しい測定値を追加
    if data_window.shape[0] < window_size:
        data_window = np.vstack([data_window, measurement])
    else:
        data_window = np.roll(data_window, -1, axis=0)
        data_window[-1] = measurement

    # ガウス畳み込みを適用して平滑化
    smoothed_data = apply_gaussian_smoothing(data_window, sigma)
    
    # 現在の平滑化された値を取得
    current_smoothed_value = smoothed_data[-1]
    
    return current_smoothed_value

# サイン波の生成
dt = 0.001  # 1stepの時間[sec]
times = np.arange(0, 1, dt)
N = times.shape[0]

f1 = 5  # サイン波の周波数1[Hz]
f2 = 10  # サイン波の周波数2[Hz]
noise_sigma = 0.5  # ノイズの分散

np.random.seed(1)
# サイン波1
x1_s = np.sin(2 * np.pi * times * f1) 
x1 = x1_s + noise_sigma * np.random.randn(N)

# サイン波2
x2_s = np.sin(2 * np.pi * times * f2) 
x2 = x2_s + noise_sigma * np.random.randn(N)

# 1000x2の配列にまとめる
data = np.column_stack((x1, x2))

# リアルタイム処理結果の保存
smoothed_x1 = []
smoothed_x2 = []

# サイン波のリアルタイム処理
for measurement in data:
    smoothed_position = process_frame(measurement)
    smoothed_x1.append(smoothed_position[0])
    smoothed_x2.append(smoothed_position[1])

# 結果のプロット
plt.figure(figsize=(12, 6))

# サイン波1のプロット
plt.subplot(2, 1, 1)
plt.plot(times, x1, label='Noisy Sine Wave 1')
plt.plot(times, x1_s, label='Original Sine Wave 1', linestyle='dashed')
plt.plot(times, smoothed_x1, label='Smoothed Sine Wave 1')
plt.title('Sine Wave 1 Smoothing')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# サイン波2のプロット
plt.subplot(2, 1, 2)
plt.plot(times, x2, label='Noisy Sine Wave 2')
plt.plot(times, x2_s, label='Original Sine Wave 2', linestyle='dashed')
plt.plot(times, smoothed_x2, label='Smoothed Sine Wave 2')
plt.title('Sine Wave 2 Smoothing')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
