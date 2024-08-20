import numpy as np

def gram_schmidt(vectors):
    """
    グラムシュミットの直交化法を用いて、与えられたベクトルの集合を直交基底に変換する。
    
    パラメータ:
    vectors (list of np.array): 入力ベクトルのリスト
    
    戻り値:
    orthogonal_vectors (list of np.array): 直交基底ベクトルのリスト
    """
    orthogonal_vectors = []
    
    for v in vectors:
        u = v.copy()
        for q in orthogonal_vectors:
            u -= np.dot(v, q) / np.dot(q, q) * q
        orthogonal_vectors.append(u / np.linalg.norm(u))
    
    return orthogonal_vectors
