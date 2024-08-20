clear

load data_new/motion_data_original2.txt
data_original = motion_data_original2;

data = zeros(size(data_original, 1), 9);
data(:,[1,2,4,5,7,8]) = data_original;

for i = 1:3
    data(:,3*i) = sqrt(1 - data_original(:,2*i-1).^2 - data_original(:,2*i).^2);
end

%%

data = data(60001:90000, :);

% フレームの間引き（重くなるのを防ぐため）
frame_skip = 1;
data = data(1:frame_skip:end, :);
num_frames = size(data, 1);

% アームのセグメントの長さ
segment_length = 1;

% 肩の位置（原点）
shoulder = [0, 0, 0];

% 図の初期化
figure;
hold on;
axis equal;
grid on;
xlim([-2, 2]);
ylim([-2, 2]);
zlim([-2, 2]);
xlabel('X');
ylabel('Y');
zlabel('Z');
view(3);

% 初期プロット
elbow = shoulder + data(1, 1:3) * segment_length;
wrist = elbow + data(1, 4:6) * segment_length;
thumb = wrist + data(1, 7:9) * segment_length;

h_shoulder_elbow = plot3([shoulder(1) elbow(1)], [shoulder(2) elbow(2)], [shoulder(3) elbow(3)], 'b', 'LineWidth', 2);
h_elbow_wrist = plot3([elbow(1) wrist(1)], [elbow(2) wrist(2)], [elbow(3) wrist(3)], 'r', 'LineWidth', 2);
h_wrist_thumb = plot3([wrist(1) thumb(1)], [wrist(2) thumb(2)], [wrist(3) thumb(3)], 'g', 'LineWidth', 2);

% フレーム番号の表示用テキストオブジェクト
h_frame_text = text(0, 6, 0, sprintf('Frame: 1'), 'FontSize', 12, 'Color', 'k');

% アニメーションループ
for i = 2:num_frames
    % 各関節の位置を計算
    elbow = shoulder + data(i, 1:3) * segment_length;
    wrist = elbow + data(i, 4:6) * segment_length;
    thumb = wrist + data(i, 7:9) * segment_length;
    
    % プロットの更新
    set(h_shoulder_elbow, 'XData', [shoulder(1) elbow(1)], 'ZData', [-shoulder(2) -elbow(2)], 'YData', [shoulder(3) elbow(3)]);
    set(h_elbow_wrist, 'XData', [elbow(1) wrist(1)], 'ZData', [-elbow(2) -wrist(2)], 'YData', [elbow(3) wrist(3)]);
    set(h_wrist_thumb, 'XData', [wrist(1) thumb(1)], 'ZData', [-wrist(2) -thumb(2)], 'YData', [wrist(3) thumb(3)]);
    
    % 本来は以下(今回はy,zが座標系として逆、zはマイナスが逆)
    % set(h_shoulder_elbow, 'XData', [shoulder(1) elbow(1)], 'YData', [shoulder(2) elbow(2)], 'ZData', [shoulder(3) elbow(3)]);
    % set(h_elbow_wrist, 'XData', [elbow(1) wrist(1)], 'YData', [elbow(2) wrist(2)], 'ZData', [elbow(3) wrist(3)]);
    % set(h_wrist_thumb, 'XData', [wrist(1) thumb(1)], 'YData', [wrist(2) thumb(2)], 'ZData', [wrist(3) thumb(3)]);
    
    set(h_frame_text, 'String', sprintf('Frame: %d', i));
    
    % 描画の更新
    drawnow;
    pause(0.03);
end

hold off;

