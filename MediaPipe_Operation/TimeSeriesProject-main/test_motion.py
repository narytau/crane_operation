import os
import cv2
import sys
import torch
import pyrealsense2 as rs
import mediapipe as mp
from torch import nn
import numpy as np
from models.model.transformer import Transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_module.BaseMotionRecognition import BaseMotionRecognition

def find_most_frequent_element(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    
    max_elements = unique_elements[counts == max_count]
    
    for elem in arr:
        if elem in max_elements:
            return elem
        
def update_judge(pred, judge):
    if not pred:
        return None  

    # 初期のjudgeを設定
    judge = pred[0]

    # 状態を追跡するための変数
    counter = [0, 0, 0, 0]  # 0から3の数の出現回数を記録

    # predの値を順に処理
    for i in range(1, len(pred)):
        current = pred[i]

        # judgeとは異なる数のカウントを増やす
        if current != judge:
            counter[current] += 1

            # 異なる数が6回中4回以上現れた場合、judgeを更新
            if counter[current] >= 4:
                judge = current
                counter = [0, 0, 0, 0]  # カウンターをリセット
                counter[judge] = 1      # 新しいjudgeのカウントを初期化

    return judge

# リアルタイムデータ取得関数の仮想例
def get_realtime_data():
    # ここにセンサーやデータストリームからデータを取得するコードを実装
    # 今はランダムに生成されたデータを使用
    data = np.random.rand(6)  # 特徴量が6つ
    return torch.tensor(data).unsqueeze(0)

# リアルタイムでモデルをテスト
def test_model_realtime(model, device, buffer_size=30):
    model.eval()
    
    # バッファを初期化
    data_buffer = []
    
    while True:
        # リアルタイムデータを取得
        data, label = get_realtime_data()
        data_buffer.append(data)
        
        # バッファが満たされたら推論を行う
        if len(data_buffer) == buffer_size:
            inputs = torch.stack(data_buffer).unsqueeze(0).to(device=device, dtype=torch.float)
            
            with torch.no_grad():
                pred = model(inputs)
            
            # バッファをリセットまたはシフト
            data_buffer = []  # 全部リセットする場合
            

def num_to_class(num):
    if num == 0:
        pred = 'High'
    elif num == 1:
        pred = 'Middle'
    elif num == 2:
        pred = 'Low'
    elif num == 3:
        pred = 'Unclassified'
    else:
        pred = 'Error'
    return pred
        



# モデルの定義と読み込み
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sequence_len = 30  # シーケンス長
max_len = 1000  # 最大シーケンス長
n_head = 4  # 注意ヘッドの数
n_layer = 2  # エンコーダーレイヤーの数
drop_prob = 0.1
d_model = 128  # 位置エンコーディング用の次元数
ffn_hidden = 512  # 分類前の隠れ層サイズ
feature = 6  # 特徴量の次元数
model = Transformer(d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, feature=feature, details=False, device=device).to(device=device)
model.load_state_dict(torch.load('myModel'))

# リアルタイムデータに対するモデルの実行
test_model_realtime(model=model, device=device)

