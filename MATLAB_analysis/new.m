load data\elbow.csv
load data\shoulder.csv
load data\wrist.csv

x_wrist = wrist(:,1);
y_wrist = wrist(:,2);
z_wrist = wrist(:,3);

x_elbow = elbow(:,1);
y_elbow = elbow(:,2);
z_elbow = elbow(:,3);

x_shoulder = shoulder(:,1);
y_shoulder = shoulder(:,2);
z_shoulder = shoulder(:,3);

wrist_list = wrist;
elbow_list = elbow;

% カルマンフィルタの初期化
% 状態ベクトル: [x, y, z, vx, vy, vz]
dt = 1 / 30; % 時間間隔（仮定）
F = [1 0 0 dt 0 0; 0 1 0 0 dt 0; 0 0 1 0 0 dt; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1];
H = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0];
Q = eye(6) * 0.01; % プロセスノイズ
R = eye(3) * 0.1; % 観測ノイズ
P = eye(6); % 共分散行列の初期値
x = [x_wrist(1); y_wrist(1); z_wrist(1); 0; 0; 0]; % 初期状態

% 過去のデータを格納する配列
filtered_wrist = zeros(500, 3);
radius_list = zeros(500,1);
angular_velocity_list = zeros(500,1);

% リアルタイム処理のループ
for t = 1:500
    % データの取得（手首、肘、肩の座標）
    wrist = [x_wrist(t), y_wrist(t), z_wrist(t)];
    elbow = [x_elbow(t), y_elbow(t), z_elbow(t)];
    shoulder = [x_shoulder(t), y_shoulder(t), z_shoulder(t)];
    
    % カルマンフィルタの予測ステップ
    x = F * x;
    P = F * P * F' + Q;

    % 観測の更新ステップ
    z = wrist'; % 観測値
    y = z - H * x;
    K = P * H' / (H * P * H' + R);
    x = x + K * y;
    P = (eye(6) - K * H) * P;

    % フィルタリングされた手首の位置を保存
    filtered_wrist(t, :) = x(1:3)';

    % 肘と手首のベクトルを計算
    wrist_to_elbow = wrist - elbow;
    elbow_to_shoulder = elbow - shoulder;

    % 楕円フィッティングのデータを更新
    if t > 1
        % 過去のデータを使って楕円フィッティング
        fitresult = fit_ellipse_3D(filtered_wrist(1:t, :));

        % 楕円のパラメータを抽出
        ellipse_center = fitresult.Center;
        ellipse_axes = fitresult.Axes;
        rotation_matrix = fitresult.RotationMatrix;

        % 半径と回転速度の計算
        long_axis = max(ellipse_axes);
        short_axis = min(ellipse_axes);
        angles = atan2(diff(filtered_wrist(1:t, 3)), diff(filtered_wrist(1:t, 1)));
        angular_velocity = diff(angles) / dt;

        % radius_list(t) = long_axis;
        % angular_velocity_list(t) = angular_velocity;

        % 判定基準に基づく円の描画の判定
        radius_threshold = 0.1; % 半径の閾値
        velocity_threshold = 0.1; % 回転速度の閾値

        % if long_axis > radius_threshold && mean(abs(angular_velocity)) > velocity_threshold
        %     disp(['円を描いていると判定されました。時間ステップ: ', num2str(t)]);
        % else
        %     disp(['円を描いていません。時間ステップ: ', num2str(t)]);
        % end
    end
end

figure(1);
time = (1:500) * dt;
t = tiledlayout(4,1);
ax1 = nexttile;
plot(ax1,time,x_wrist,"LineWidth",1.5); hold on
plot(ax1,time,filtered_wrist(:,1),"LineWidth",1.5);
ylabel('x');
legend('True', 'Filtered');

ax2 = nexttile;
plot(ax2,time,y_wrist,"LineWidth",1.5); hold on
plot(ax2,time,filtered_wrist(:,2),"LineWidth",1.5);
ylabel('y');
legend('True', 'Filtered');

ax3 = nexttile;
plot(ax3,time,z_wrist,"LineWidth",1.5); hold on
plot(ax3,time,filtered_wrist(:,3),"LineWidth",1.5);
ylabel('z');
legend('True', 'Filtered');

ax4 = nexttile;
plot(ax4,time,[0;sqrt(sum((diff(filtered_wrist)).^2, 2))],"LineWidth",1.5); 
ylabel('velocity');

figure(2);
plot3(filtered_wrist(:,1),filtered_wrist(:,2),filtered_wrist(:,3),'LineWidth',1.5);
xlabel('x'); ylabel('y'); zlabel('z');


% fit_ellipse_3D 関数の定義
function [fitresult] = fit_ellipse_3D(data)
    % データの中心化
    data_mean = mean(data, 1);
    centered_data = data - data_mean;

    % 共分散行列の計算
    covariance_matrix = cov(centered_data);

    % 共分散行列の固有値と固有ベクトルの計算
    [eigen_vectors, eigen_values] = eig(covariance_matrix);

    % 固有値の平方根を取ることで楕円の軸の長さを取得
    axes_lengths = sqrt(diag(eigen_values));

    % 結果を構造体に格納
    fitresult.Center = data_mean;
    fitresult.Axes = axes_lengths;
    fitresult.RotationMatrix = eigen_vectors;
end
