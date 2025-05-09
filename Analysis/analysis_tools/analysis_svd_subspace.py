
import os
import torch
import numpy as np

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


# 特異値分解を実行
def svd(opt, features, labels, plot=True, max_k=20, cls_per_task=1, name="practice", threshold=0.9, mode="task"):
    """
    features : [num_data, embed_dim] の tensor
    labels   : [num_data] の tensor
    mode     : "task" | "cls" | "all"
    """

    # 特徴量の正規化（追加）
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    print("features.shape: ", features.shape)

    if mode == "all":
        task_feats = [features]
        task_labs  = [labels]
        cls_sets   = [sorted(labels.unique().tolist())]
        cls_or_task = "all"
    else:
        task_feats, task_labs, cls_sets = split_by_task(features, labels, cls_per_task)
        cls_or_task = "cls" if cls_per_task == 1 else "task"
    

    plt.figure()
    label_list = []
    k_list = []

    svd_dict = {}

    # タスクorクラス毎にデータを取り出してSVDを実行
    for t, (z, y) in enumerate(zip(task_feats, task_labs)):
        print(f"{cls_or_task.title()} {t} → classes {cls_sets[t]}")
        label_list.append(t)

        # 中心化
        z_centered = z - z.mean(dim=0, keepdim=True)

        # 共分散行列
        cov = z_centered.T @ z_centered / (z.size(0) - 1)
        # print("cov.shape: ", cov.shape)   # cov.shape:  torch.Size([512, 512])

        # SVD
        U, S, V = torch.svd(cov)
        # print("U.shape: ", U.shape)   # U.shape:  torch.Size([512, 512])
        # print("S.shape: ", S.shape)   # S.shape:  torch.Size([512])
        # print("V.shape: ", V.shape)   # V.shape:  torch.Size([512, 512])


        # SVDの結果を確認（UとVは同値＝固有値分解の固有値ベクトル）
        # cos = torch.nn.CosineSimilarity(dim=0)
        # cos_similarities = cos(U, V)
        # print("cos_similarities: ", cos_similarities)
        # print("cos_similarities.shape: ", cos_similarities.shape)

        svd_dict[t] = [U, S, V]
    
    return svd_dict
    

# 同一タスク・クラスのデータに対する特徴表現のsub space類似度を計算
def subspace_similarity(svd_dict1, svd_dict2, threshold):

    common_keys = set(svd_dict1.keys()) & set(svd_dict2.keys())
    print("common_keys: ",common_keys)

    dist_dict = {}

    # 共通データに対するタスクid=t, t+1のモデルのsub spaceの類似度を計算
    # key=1であれば，タスク1のデータに対するsub spaceの類似度
    for key in common_keys:
        
        """
        key（=使用データのラベルorタスクid）
        """

        U1, S1, V1 = svd_dict1[key]
        U2, S2, V2 = svd_dict2[key]

        # 累積寄与率 α_k の計算
        total_var = S1.sum()
        alpha_k1 = torch.cumsum(S1, dim=0) / total_var
        total_var = S2.sum()
        alpha_k2 = torch.cumsum(S2, dim=0) / total_var

        # 寄与率しきい値を満たす最小次元k
        k_dim1 = int((alpha_k1 > threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        k_dim2 = int((alpha_k2 > threshold).nonzero(as_tuple=True)[0][0].item()) + 1

        # 最大となる累積寄与率 α を選択
        k = max(k_dim1, k_dim2)
        # print("k: ", k)  # k:  228

        # 上位特異値に対応するベクトルのみ取り出す
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]
        # print("U1_k.shape: ", U1_k.shape)   # U1_k.shape:  torch.Size([512, 228])
        # print("U2_k.shape: ", U2_k.shape)   # U2_k.shape:  torch.Size([512, 228])

        # # 部分空間 (sub space) を計算
        # P1 = U1_k @ U1_k.T
        # P2 = U2_k @ U2_k.T

        # # Frobeniusノルムの計算
        # diff = P1 - P2
        # dist = torch.norm(diff, p='fro')

        diff = U1_k.T @ U2_k
        # print("diff: ", diff)
        dist = torch.norm(diff, p='fro') ** 2
        print("dist: ", dist)
        dist = dist / k
        
        print("debug")
        print(key, dist)

        dist_dict[key] = dist

    return dist_dict



# 異なるタスクのデータの特徴表現に対するsub spaceの類似度を計算
def subspace_similarity_diff_task(svd1, svd_dict2, threshold, name="practice", opt=None):

    keys = svd_dict2.keys()

    dist_dict = {}

    for key in keys:

        U1, S1, V1 = svd1
        U2, S2, V2 = svd_dict2[key]
        print("U1.shape: ", U1.shape)

        # 累積寄与率 α_k の計算
        total_var = S1.sum()
        alpha_k1 = torch.cumsum(S1, dim=0) / total_var
        total_var = S2.sum()
        alpha_k2 = torch.cumsum(S2, dim=0) / total_var

        # 寄与率しきい値を満たす最小次元k
        k_dim1 = int((alpha_k1 > threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        k_dim2 = int((alpha_k2 > threshold).nonzero(as_tuple=True)[0][0].item()) + 1

        # 最大となる累積寄与率 α を選択
        # k = max(k_dim1, k_dim2)
        k = k_dim1
        print("k: ", k)  # k:  228

        # 上位特異値に対応するベクトルのみ取り出す
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]
        # print("U1_k.shape: ", U1_k.shape)   # U1_k.shape:  torch.Size([512, 228])
        # print("U2_k.shape: ", U2_k.shape)   # U2_k.shape:  torch.Size([512, 228])

        # # 部分空間 (sub space) を計算
        # P1 = U1_k @ U1_k.T
        # P2 = U2_k @ U2_k.T

        # # Frobeniusノルムの計算
        # diff = P1 - P2
        # dist = torch.norm(diff, p='fro')

        diff = U1_k.T @ U2_k
        dist = torch.norm(diff, p='fro') ** 2
        dist = dist / k
        
        print(key, dist)

        dist_dict[key] = dist



    return dist_dict



def plot_diff_dist(diff_dist, opt, name=None):

    # 全てのクラスID (key) を列挙
    all_keys = sorted({k.item() if hasattr(k, 'item') else k 
                    for inner in diff_dist.values() for k in inner.keys()})
    task_ids = sorted(diff_dist.keys())

    # 各クラスIDごとに時系列で距離値を取得
    classwise_dists = {k: [] for k in all_keys}

    for t in task_ids:
        for k in all_keys:
            v = diff_dist[t].get(k, None)
            if v is not None and hasattr(v, 'item'):
                classwise_dists[k].append(v.item())
            elif v is not None:
                classwise_dists[k].append(v)
            else:
                classwise_dists[k].append(None)

    # プロット
    plt.figure(figsize=(10, 6))
    for k, values in classwise_dists.items():
        plt.plot(task_ids, values, label=f'Class {k}', marker='o')

    plt.xlabel('Target Task t')
    plt.ylabel('Projection Distance')
    plt.title(f'{opt.method}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{opt.save_path}/{name}_{opt.method}.pdf")

    plt.show()


    return None



