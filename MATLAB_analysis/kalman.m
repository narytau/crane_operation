clear


load data/motion_data_30.txt
data = motion_data_30(1:1500,:);

position_size = size(data,2);

%% カルマンフィルタの初期化
% 状態ベクトル: [x, y, z, vx, vy, vz]
dt = 1 / 30; % 時間間隔
EYE = eye(position_size);
ZEROS = zeros(position_size);
F = [EYE, EYE * dt; ZEROS, EYE];
H = [EYE, ZEROS];
Q = EYE * 0.01; % プロセスノイズ
R = EYE * 0.1; % 観測ノイズ
P = EYE; % 共分散行列の初期値
x = [data(1,:)' ; zeros(position_size,1)]; % 初期状態

filtered_data = zeros(size(data));

%% リアルタイム処理のループ
for t = 1:size(data,1)
    % カルマンフィルタの予測ステップ
    x = F * x;
    P = F * P * F' + Q;

    % 観測の更新ステップ
    z = wrist'; % 観測値
    y = z - H * x;
    K = P * H' / (H * P * H' + R);
    x = x + K * y;
    P = (eye(position_size) - K * H) * P;

    % フィルタリングされた手首の位置を保存
    filtered_data(t, :) = x(1:position_size)';
end