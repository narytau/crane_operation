clear

load Measurement/array_10.txt
load Measurement/array_20.txt
load Measurement/array_30.txt
load Measurement/array_40.txt
load Measurement/array_50.txt

std_array = zeros(5, 3);
std_array(1,:) = std(array_10);
std_array(2,:) = std(array_20);
std_array(3,:) = std(array_30);
std_array(4,:) = std(array_40);
std_array(5,:) = std(array_50);

dist_array = zeros(5, 1);
dist_array(1) = std(sqrt(sum(array_10.^2, 2)));
dist_array(2) = std(sqrt(sum(array_20.^2, 2)));
dist_array(3) = std(sqrt(sum(array_30.^2, 2)));
dist_array(4) = std(sqrt(sum(array_40.^2, 2)));
dist_array(5) = std(sqrt(sum(array_50.^2, 2)));

figure(1);
plot(10:10:50, std_array, 'LineWidth',2);
legend('x', 'y', 'z');

figure(2);
plot(10:10:50, dist_array, 'LineWidth',2);
