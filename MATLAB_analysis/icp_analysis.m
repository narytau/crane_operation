%%
clear;
load data_new/pose_matrix_iphone.txt


data = zeros(30, 3, 6);
for i = 1:6
    data(:,:,i) = pose_matrix_iphone(:,3*(i-1)+1:3*i);
end

for i = 1:3
    data(:,i,:) = data(:,i,:) - data(:,i,3);
end

data = data(:, :, [1,4,5,6]);

X = zeros(3);
for i = 1:3

X(:,i) = pinv(permute(data(:,i,[2,3,4]), [1,3,2])) * data(:,i,1);



end

permute(data(:,2,[2,3,4]), [1,3,2]) * X(:,2) - data(:,2,1)
X