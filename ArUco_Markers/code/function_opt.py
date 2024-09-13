import numpy as np
from scipy.optimize import minimize

def penalty_for_orthogonality(R_flat):
    R = R_flat.reshape(3, 3)
    ortho_diff = R.T @ R - np.eye(3)
    return np.sum(ortho_diff ** 2)  # 直交性からのずれをペナルティとして計算

# 目的関数
def objective_with_penalty(R_flat, T, position_init, p_extracted,  penalty_weight=1e+8):
    R = R_flat.reshape(3, 3)
    N = position_init.shape[0]
    func = R @ position_init.T - np.tile(T.reshape(3,1), N) - p_extracted.T
    # 目的関数 + 直交性ペナルティ
    return np.sum(func ** 2) + penalty_weight * penalty_for_orthogonality(R_flat)

# 行列式の制約: det(R) = 1
def determinant_constraint(R_flat):
    R = R_flat.reshape(3, 3)
    return np.linalg.det(R) - 1

def calc_angle(vec1, vec2):
    cos_theta = np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    theta = np.arccos(cos_theta) * 180 / np.pi
    return theta

def arange_position_init(position):
    """coordinate transformation

    Args:
        position (Tags_num x 3):
    """
    point_num = position.shape[0]
    position_diff = position - position[0, :]

    # theta, length
    relative_array = np.zeros((point_num-2, 2))
    for i in range(point_num-2):
        base_length = np.linalg.norm(position_diff[1,:])
        relative_array[i, 0] = calc_angle(position_diff[1,:], position_diff[i+2,:])
        relative_array[i, 1] = np.linalg.norm(position_diff[i+2,:]) / base_length

    # position_init
    position_init = np.zeros((point_num, 3))
    position_init[1, :] = np.array([np.linalg.norm(position_diff[1,:]), 0, 0])
    for i in range(point_num-2):
        theta = relative_array[i, 0] * np.pi / 180
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta),  0],
                                    [0, 0, 1]])
        position_init[i+2, :] =  relative_array[i, 1] * rotation_matrix @ position_init[1, :] 

    # Regularization for inverse matrix computation
    position_init[:, -1] = 1e-6

    return position_init



p1 = np.array([[ 0.09402263,  0.00970577,  0.35131145],
            [-0.0639756,   0.05762789,  0.33413473],
            [ 0.06881183, -0.09454012,  0.37391877],
            [-0.10946034, -0.03037025,  0.35205835],
            [-0.00158597, -0.03285263,  0.35782278]])

p2 = np.array([[ 0.1192032 ,  0.02870564,  0.40588641],
            [-0.00283386,  0.05382122,  0.2964115 ],
            [ 0.09044836, -0.07619598,  0.42913172],
            [-0.04653129, -0.03736651,  0.29832276],
            [ 0.03919235, -0.02506492,  0.36430935]])

p3 = np.array([[-3.03031834e-04, -6.87972813e-02, 3.76999690e-01],
            [ 1.47533403e-01,  8.84732811e-03,  3.75527033e-01],
            [-5.84643755e-02,  2.49697399e-02,  3.79557584e-01],
            [ 1.16605030e-01,  1.05664750e-01,  3.80921390e-01],
            [ 3.67811703e-02,  3.00543562e-02,  3.76889766e-01]])

#0.006誤差
p1_ = p1 - p1[0, :]
p2_ = p2 - p2[0, :]
p3_ = p3 - p3[0, :]

point_num = 5



position_init = arange_position_init(position=p1)

# Intitialization
R = np.random.rand(3,3)
print(R)    

R_tmp = np.ones((3, 3)) * 1e+10

threshold = 1e-20

T = np.zeros(3)

p_extracted = p2_[1:, :]

N = point_num - 1
i=0

# 制約条件を定義
constraints = [
    {'type': 'eq', 'fun': determinant_constraint}     # 行列式の制約
]

for iter in range(100):
    # Optimal T
    T = np.sum(p_extracted.T - R @ position_init[1:,:].T, axis=1) / N

    # 最適化の実行
    result = minimize(objective_with_penalty, R.flatten(), args=(T, position_init[1:,:], p_extracted),
                    constraints=constraints,
                    method='SLSQP')

    # 結果の行列
    R = result.x.reshape(3, 3)

    # Stop iteration when R does not change
    if np.sum((R_tmp - R)**2) < threshold:
        print(iter)
        break
    
    if iter == 100:
        print("Cannot clculate")
    
    R_tmp = R
    
print(R)
print("check", R.T@R, np.linalg.det(R), objective_with_penalty(R.flatten(), T, position_init[1:,:], p_extracted))