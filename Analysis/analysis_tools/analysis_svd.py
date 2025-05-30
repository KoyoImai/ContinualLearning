import os
import torch

import matplotlib.pyplot as plt


"""
クラス・タスクなどの粒度を変えながらSVDしたい
"""



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
    # labels = labels.cpu()
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
        # mask = (task_id == t).cpu()
        mask = (task_id == t)
        if mask.sum() == 0:
            continue
        task_features.append(features[mask])   # [n_t, D]
        task_labels.append(labels[mask])       # [n_t]

    return task_features, task_labels, label_list4task




def svd(opt, features, labels, plot=True, max_k=20, cls_per_task=1, name="practice", threshold=0.9, mode="task", use_gpu=True, use_num_data=None):
    """
    features : [num_data, embed_dim] の tensor
    labels   : [num_data] の tensor
    mode     : "task" | "cls" | "all"
    """
    
    # 特徴量の正規化（追加）
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    # features = torch.nn.functional.normalize(features, p=2, dim=1).cpu()

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
    k_dict = {}

    for t, (z, y) in enumerate(zip(task_feats, task_labs)):
        print(f"{cls_or_task.title()} {t} → classes {cls_sets[t]}")
        label_list.append(t)

        if not use_gpu:
            z = z.cpu()
        # if use_gpu and use_num_data is not None:
        if use_num_data is not None:
            # 各クラスのデータの特徴量を use_num_data だけ取り出す
            # y を確認して，どのようなラベルがあるかを確認
            # 確認したラベルの特徴量が当分（use_num_data個）になるように取り出す
            # zの形式は元のまま

            z_new_list = []
            y_new_list = []

            unique_labels = y.unique()
            for lbl in unique_labels:
                idx = (y == lbl).nonzero(as_tuple=True)[0]
                if idx.size(0) < use_num_data:
                    continue  # スキップ or 警告出して終了してもよい
                selected = idx[torch.randperm(len(idx))[:use_num_data]]
                selected = selected.to(z.device)
                z_new_list.append(z[selected])
                y_new_list.append(y[selected])

            z = torch.cat(z_new_list, dim=0)
            y = torch.cat(y_new_list, dim=0)

        print("z.shape: ", z.shape)

        # 中心化
        z_centered = z - z.mean(dim=0, keepdim=True)

        # 共分散行列
        cov = z_centered.T @ z_centered / (z.size(0) - 1)
        # print("cov.shape: ", cov.shape)   # cov.shape:  torch.Size([512, 512])

        # SVD
        U, S, V = torch.svd(cov)
        # U, S, V = torch.svd(cov.cpu())


        # 累積寄与率 α_k の計算
        total_var = S.sum()
        alpha_k = torch.cumsum(S, dim=0) / total_var

        # 寄与率しきい値を満たす最小次元k
        k_dim = int((alpha_k > threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        print(f"  - α_k > {threshold:.2f} となる最小k: {k_dim}")
        k_list.append(k_dim)
        k_dict[f"{cls_or_task}{t}"] = k_dim

        if plot:
            k = min(max_k, len(alpha_k))
            plt.plot(range(1, k+1), alpha_k[:k].cpu().numpy(), label=f'{cls_or_task} {t} (k={k_dim})')
        

        # # === 新機能：正規化済みの非ゼロ特異値の累積和プロット ===
        # nonzero_singular_values = S[S > 0]
        # cumulative_sum = torch.cumsum(nonzero_singular_values, dim=0)
        # normalized_cumsum = (cumulative_sum / cumulative_sum[-1]).cpu().numpy()

        # plt.figure()
        # plt.plot(range(1, len(normalized_cumsum) + 1), normalized_cumsum, label=f'Normalized cumulative sum of S')
        # plt.xlabel('Index of singular value')
        # plt.ylabel('Normalized cumulative sum')
        # plt.title(f'Normalized Non-zero Singular Value Accumulation ({cls_or_task} {t})')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()

        # if opt.save_path:
        #     save_dir = f"{opt.save_path}/{name}_{mode}_projector/cumsum/" if opt.projector else f"{opt.save_path}/{name}/cumsum/"
        #     os.makedirs(save_dir, exist_ok=True)
        #     file_path = f"{save_dir}/model{opt.target_task}_{cls_or_task}_{t}_cumsum.pdf"
        #     plt.savefig(file_path)

        # plt.show() if plot else plt.clf()
        # # ====================================================
        

        del z, z_centered, cov, U, S, V
        torch.cuda.empty_cache()  # 明示的にメモリを開放

    if plot:
        plt.xlabel('number of k')
        plt.ylabel('alpha_k')
        plt.xlim([0, 20.05])
        plt.ylim([0, 1.05])
        plt.title(f'{opt.method}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if opt.save_path:
            save_dir = f"{opt.save_path}/{name}_{mode}_projector/" if opt.projector else f"{opt.save_path}/{name}/"
            os.makedirs(save_dir, exist_ok=True)
            file_path = f"{save_dir}/model{opt.target_task}_{cls_or_task}_{label_list[0]}to{label_list[-1]}.pdf"
            plt.savefig(file_path)

        plt.show() if plot else plt.clf()

    return None



def svd_all_layers(opt, layer_outputs: dict, labels: torch.Tensor, mode="task", cls_per_task=None, name="practice"):
    """
    各層の出力に対してSVDを実行するユーティリティ

    Parameters:
    -----------
    layer_outputs : dict
        {layer_name: feature_tensor [N, D]} の辞書
    labels : torch.Tensor
        全データのラベル [N]
    mode : str
        "task" | "cls" | "all" のいずれか（既存svd関数のmodeに対応）
    cls_per_task : int or None
        mode="task"のときに必要

    Returns:
    --------
    layer_k_dict : dict
        各層のlayer_nameをkey、rank (有効次元数) のdict
    """
    all_k = {}

    for name, features in layer_outputs.items():
        print(f"\n===== SVD: Layer = {name} =====")
        # フィーチャごとに個別SVD分析を実行
        svd(opt=opt,
            features=features,
            labels=labels,
            cls_per_task=opt.cls_per_task,
            name=f"{opt.block_type}_{opt.flatten_type}/layer_{name}",
            use_gpu=True,
            # use_num_data=10
            )
        
        # results = svd(opt=opt,
        #               features=features,
        #               labels=labels,
        #               cls_per_task=opt.cls_per_task,
        #               name="alldata")

    return all_k  




# 引数taskを未実装
# def svd(opt, features, labels, plot=True, max_k=20, cls_per_task=1, name="practice", threshold=0.9):
    
#     """
#         features:[num_data, embed_dim]のtensor
#         labels  :[num_data]のtensor
#     """
    
#     task_feats, task_labs, cls_sets = split_by_task(features, labels, cls_per_task)
    
#     if cls_per_task == 1:
#         cls_or_task = "cls"
#     else:
#         cls_or_task = "task"

#     plt.figure()

#     label_list = []
#     k_list = []
#     for t, (z, y) in enumerate(zip(task_feats, task_labs)):
        
#         # --- ここで SVD / intrinsic_dim など ----
#         print(f"Task {t} → classes {cls_sets[t]}")
        
#         label_list.append(t)

#         # 中心化
#         z_centered = z - z.mean(dim=0, keepdim=True)

#         # 共分散行列
#         cov = z_centered.T @ z_centered / (z.size(0) - 1)  # [d, d]

#         # 共分散行列に対してSVD
#         U, S, V = torch.svd(cov)  # Sは固有値と一致

#         # 累積寄与率 α_k の計算
#         total_var = S.sum()
#         alpha_k = torch.cumsum(S, dim=0) / total_var

#         k_dim = int((alpha_k > threshold).nonzero(as_tuple=True)[0][0].item()) + 1
#         print(f"  - α_k > {threshold:.2f} となる最小k: {k_dim}")
#         k_list.append(k_dim)

#         # プロット
#         if plot:
#             k = min(max_k, len(alpha_k))
#             plt.plot(range(1, k+1), alpha_k[:k].cpu().numpy(), label=f'{cls_or_task} {t} (k={k_dim})')
    
#     if plot:

#         plt.xlabel('number of k')
#         plt.ylabel('alpha_k')

#         plt.xlim([0, 20.05])
#         plt.ylim([0, 1+0.05])

#         plt.title(f'{opt.method}')
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()

#         if opt.save_path:
#             if opt.projector:
#                 save_path = f"{opt.save_path}/{name}_projector/"
#             else:
#                 save_path = f"{opt.save_path}/{name}/"
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             file_path = f"{save_path}/model{opt.target_task}_{cls_or_task}{label_list}.pdf"
#             plt.savefig(file_path)
        
#         if plot:
#             # plt.xlim([0, 1.0])
#             # plt.ylim([0, max_k+0.5])
#             plt.show()
#         else:
#             plt.clf()

#     return None











# # クラス毎に特異値分解を実行
# def svd(opt, features, labels, plot=True, max_k=20, cls_per_task=1):
    
#     """
#         features:[num_data, embed_dim]のtensor
#         labels  :[num_data]のtensor
#     """
    
#     # クラスリストの作成
#     label_list = sorted(torch.unique(labels).tolist())

#     # タスク毎のクラスリスト(opt.cls_per_taskを使いたい)
#     task_id = labels // opt.cls_per_task
#     label_list4task = []

#     plt.figure()

#     for label in label_list:

#         index = (labels == label).nonzero(as_tuple=True)[0]
#         # print("index: ", index)
        
#         z = features[index]  # [n, d]

#         if z.size(0) < 2:
#             continue  # サンプル数が1以下のクラスはスキップ

#         # 中心化
#         z_centered = z - z.mean(dim=0, keepdim=True)
#         # print("z_centered.shape: ", z_centered.shape)

#         cov = z_centered.T @ z_centered / (z.size(0) - 1)  # [d, d]
#         # print("cov.shape: ", cov.shape)

#         # 共分散行列に対してSVD
#         U, S, V = torch.svd(cov)  # Sは固有値と一致

#         # 累積寄与率 α_k の計算
#         total_var = S.sum()
#         alpha_k = torch.cumsum(S, dim=0) / total_var

#         # プロット
#         if plot:
#             k = min(max_k, len(alpha_k))
#             plt.plot(range(1, k+1), alpha_k[:k].cpu().numpy(), label=f'class {label}')
    
#     if plot:

#         plt.xlabel('k')
#         plt.ylabel('alpha_k')
#         plt.title('cov SVD alpha_k')
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()

#         if opt.save_path:
#             os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)
#             file_path = f"{opt.save_path}/model{opt.target_task}_label{label_list}.pdf"
#             plt.savefig(file_path)
        
#         if plot:
#             plt.show()
#         else:
#             plt.clf()

#     return None