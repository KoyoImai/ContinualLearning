
import os

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




# タスク毎に特徴量，ラベルを分割
def split_by_task(features: torch.Tensor,
                  labels: torch.Tensor,
                  cls_per_task: int):
    """
    features : [N, D]  – 抽出済み特徴
    labels   : [N]     – 0,1,2,... のグローバルクラス ID
    cls_per_task : int – 1タスクに含まれるクラス数
    -------------------------------------------------
    戻り値:
        task_features : List[Tensor]  # タスク t の全特徴行列
        task_labels   : List[Tensor]  # 〃 に対応するラベル
        label_list4task : List[List[int]]  # 各タスクのクラス ID
    """
    # ---------- タスク ID を計算 ---------------
    # “クラス ID // cls_per_task” がそのサンプルのタスク番号になる
    task_id = labels // cls_per_task          # [N]
    num_tasks = int(task_id.max().item()) + 1

    task_features, task_labels, label_list4task = [], [], []

    for t in range(num_tasks):
        # ① そのタスクに属するクラス集合
        cls_start = t * cls_per_task
        cls_end   = (t + 1) * cls_per_task
        class_ids = list(range(cls_start, cls_end))
        label_list4task.append(class_ids)

        # ② マスクして特徴を抽出
        mask = (task_id == t)
        if mask.sum() == 0:
            continue
        task_features.append(features[mask])   # [n_t, D]
        task_labels.append(labels[mask])       # [n_t]

    return task_features, task_labels, label_list4task


def compute_pca(opt, features, labels, n_components=20, cls_per_task=1):

    task_feats, task_labs, cls_sets = split_by_task(features, labels, cls_per_task)
    cls_or_task = "cls" if cls_per_task == 1 else "task"

    label_list = []
    k_list = []

    pca_dict = {}
    pca_list = []

    # クラスorタスク毎に特徴量を取り出してPCA
    for t, (z, y) in enumerate(zip(task_feats, task_labs)):
        print(f"{cls_or_task.title()} {t} → classes {cls_sets[t]}")
        label_list.append(t)

        # 中心化
        z_centered = z - z.mean(dim=0, keepdim=True)
        z_centered = z_centered.cpu().numpy()

        # PCAを実行
        pca = PCA(n_components=n_components)
        pca.fit(z_centered)
        print("pca.components_.shape: ", pca.components_.shape)

        pca_dict[t] = pca.components_
        pca_list.append(pca.components_)

    return pca_dict, pca_list





def compute_mean_drift(pca_dict1, pca_dict2, opt=None):
    """
    2つのpca辞書（class_id or task_id → [n_components, D]）間のdriftを計算
    """
    common_keys = set(pca_dict1.keys()) & set(pca_dict2.keys())
    drift_values = []

    for key in common_keys:
        U1 = pca_dict1[key]  # shape: [k, D]
        U2 = pca_dict2[key]

        # 各主成分ベクトルのcos類似度（内積の絶対値）
        cos_sims = np.abs(np.sum(U1 * U2, axis=1))  # shape: [k]
        drift = 1.0 - np.mean(cos_sims)
        drift_values.append(drift)

    if drift_values:
        return float(np.mean(drift_values))
    else:
        return None