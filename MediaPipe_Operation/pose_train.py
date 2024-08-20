import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def normalize_angle(angle):
    """Function to normalize an angle to the range [-3pi/4, 5pi/4]"""
    origin = 110 * np.pi / 180
    return (angle + origin) % (2 * np.pi) - origin

base_path = 'C:\\Users\\nakamura\\Downloads\\Stuttgart_git\\MediaPipe_Operation\\theta_data\\'
loaded_theta_data = np.loadtxt(base_path + 'theta_data_ver2.txt')
original_shape = (int(len(loaded_theta_data)/3/4), 3, 4)
loaded_theta_data = loaded_theta_data.reshape(original_shape)

theta_data = loaded_theta_data[:,:,0]
for i in range(loaded_theta_data.shape[-1]-1):
    theta_data = np.concatenate([theta_data, loaded_theta_data[:,:,i+1]])

theta_data[[22,28,29],0] -= 360
theta_data[29, :] = [-92, -86, -81]

labels = np.array(['Up'] * original_shape[0] +
                  ['Down'] * original_shape[0] + 
                  ['Right'] * original_shape[0] + 
                  ['Left'] * original_shape[0])

scaler = StandardScaler()
theta_scaled = scaler.fit_transform(theta_data)
pickle.dump(scaler, open(base_path+'scaler.sav', 'wb'))

# test_samples_scaled = scaler.transform(test_samples)

# SVMモデルの作成とトレーニング
svm_model = SVC(probability=True)  # probability=Trueで確率を出力できるようにする
svm_model.fit(theta_scaled, labels)

# Save
with open(base_path+'model2.pickle', mode='wb') as f:
    pickle.dump(svm_model, f, protocol=2)

for i in range(loaded_theta_data.shape[-1]):
    plt.scatter(loaded_theta_data[:,2,i], loaded_theta_data[:,1,i])
plt.legend(['Up','Down','Right','Left'])
plt.show()