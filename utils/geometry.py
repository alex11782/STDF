import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def compute_laplacian(verts, faces):
    """计算简化的图拉普拉斯矩阵"""
    V = verts.shape[0]
    # 构建邻接矩阵
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones_like(rows)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(V, V))
    adj = (adj + adj.T) > 0

    # 度矩阵
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - adj
    return L.tocoo(), degrees


def get_operators(verts_np, faces_np, k_eig=128, device='cuda'):
    """预计算谱算子"""
    L_coo, mass_np = compute_laplacian(verts_np, faces_np)

    # 特征分解
    sigma = 1e-8
    try:
        evals, evecs = sla.eigsh(L_coo, k=k_eig, M=sp.diags(mass_np), sigma=sigma, which='LM')
    except:
        evals, evecs = sla.eigsh(L_coo, k=k_eig, sigma=sigma, which='LM')

    evals = torch.from_numpy(evals.astype(np.float32)).to(device)
    evecs = torch.from_numpy(evecs.astype(np.float32)).to(device)
    # mass 在这里仅作占位，简单起见设为1
    mass = torch.ones(verts_np.shape[0], device=device)

    return evals, evecs, mass