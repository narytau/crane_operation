## 必要な環境
RealSenseカメラ
Python 3.x
必要なPythonパッケージ（opencv, pyrealsense2, mediapipeなど）
## フォルダ構成
pose_aruco_realsense_pseudo.py
RealSenseカメラを使用して壁面に取り付けられたArUcoタグを識別し、カメラ座標系と壁座標系の関係を得るコードが含まれています。
MediaPipe_Operation
クレーン信号手の指示をカメラで分析するためのコードおよびモデルが格納されています。
## 使用方法
必要なライブラリをインストールします。
pose_aruco_realsense_pseudo.pyを使用して、カメラと壁面の座標系の関係を取得します。
MATLAB_analysisで骨格位置変化の分析を行います。
MediaPipe_Operationで指示をカメラで認識します。
