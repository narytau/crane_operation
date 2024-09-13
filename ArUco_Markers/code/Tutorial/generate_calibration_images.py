import pyrealsense2 as rs
import numpy as np
import cv2
import os

# デバイスの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ストリームの開始
pipeline.start(config)

# 保存するディレクトリを設定
save_dir = 'images/calib_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# カウンターの初期化
counter = 1

print("Press 'Tab' to capture a photo. Press 'q' to exit.")

try:
    while True:
        # フレームを取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # 画像を取得
        color_image = np.asanyarray(color_frame.get_data())

        # 画像を表示
        cv2.imshow('RealSense', color_image)

        # キーのキャプチャ
        key = cv2.waitKey(1) & 0xFF
        if key == 9:  # TabキーのASCIIコード
            # 画像の保存
            image_path = os.path.join(save_dir, f'calibration_{counter}.jpg')
            cv2.imwrite(image_path, color_image)
            print(f"Image saved: {image_path}")
            counter += 1
        elif key == ord('q'):
            # 'q'キーが押されたら終了
            break
        
finally:
    # ストリームの停止
    pipeline.stop()
    cv2.destroyAllWindows()
