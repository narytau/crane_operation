clear

load data_new/motion_data_original7.txt
load data_new/motion_data_unclass11.txt
load data_new/motion_data_low_exp.txt

data_low_exp = motion_data_low_exp;
data_original = motion_data_original7;
data_unclass = motion_data_unclass11;

data_size = size(data_original);
data_middle = data_original(data_size(1)/3+1:end, :);
data_low = data_original(data_size(1)*2/3+1:end, :);

figure(1);
plot(data_low(360:720, :), LineWidth=2);
legend('1','2','3','4','5','6');

figure(2);
plot(data_unclass(360:720, :), LineWidth=2);
legend('1','2','3','4','5','6');

figure(3);
plot(data_low_exp(:, :), LineWidth=2);
legend('1','2','3','4','5','6');

% middle 360s -> 9 times
% 360s -> 10 times
% 36s  -> 1 time
