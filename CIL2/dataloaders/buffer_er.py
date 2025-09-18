import random
import math
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset

from dataloaders.tiny_imagenets import TinyImagenet



def set_replay_samples_reservoir(opt, model, prev_indices=None):

    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # データセットの仮作成（ラベルがほしいだけ）
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'cifar100':
        subset_indices = []
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = val_dataset.targets

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    # 前回タスクのクラスを獲得
    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
    else:
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices

    # 前回タスクのデータのインデックス獲得
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()
    
    total_indices = prev_indices + observed_indices
    # print("1 total_indices: ", total_indices)
    
    # ランダムにバッファサイズ分だけ取り出す
    random.shuffle(total_indices)
    # print("2 total_indices: ", total_indices)
    # assert False

    total_indices = total_indices[:opt.mem_size]

    return total_indices



# ring buffer
def set_replay_samples_ring(opt, model, prev_indices=None):

    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # データローダの仮作成（ラベルがほしいだけ）
    val_transform = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'cifar100':
        subset_indices = []
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = val_dataset.targets

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    

    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
    else:

        # 過去タスクのデータに割り当てるバッファのサイズ
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)

        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                # 各クラス均等になるようにバッファ内のデータを削除
                prev_indices += torch.tensor(_prev_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))

        # 前回タスクのクラス範囲
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))
    
    print("buffer_er.py observed_classes: ", observed_classes)

    # 確認済みのクラス（前回タスク）がない場合終了
    if len(observed_classes) == 0:
        return prev_indices
    

    # 確認済みクラスのインデックスを獲得
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()


    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)
    print("val_unique_cls: ", val_unique_cls)


    print("opt.mem_size: ", opt.mem_size)
    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        print("size_for_c_flaot: ", size_for_c_float)
        p = size_for_c_float -  ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        mask = val_targets[observed_indices] == c
        selected_observed_indices += torch.tensor(observed_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()
    print(np.unique(val_targets[selected_observed_indices], return_counts=True))


    model.is_training = is_training

    return prev_indices + selected_observed_indices






import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import torch.nn.functional as F

# =========================================
# k-means（PyTorch簡易実装）
# =========================================
def kmeans_torch(X, k, max_iters=50, tol=1e-4):
    """
    X: (N, D) tensor
    k: クラスタ数
    """
    N = X.size(0)
    if k >= N:
        # kがサンプル数以上なら各サンプルをそのまま中心とする
        centers = X.clone()
        assign = torch.arange(N, device=X.device)
        return centers, assign

    # 初期中心はランダムに選択
    perm = torch.randperm(N, device=X.device)
    centers = X[perm[:k]].clone()

    prev_assign = None
    for _ in range(max_iters):
        # 距離計算（L2距離の2乗）
        x_norm = (X * X).sum(dim=1, keepdim=True)          # (N,1)
        c_norm = (centers * centers).sum(dim=1, keepdim=True).T  # (1,k)
        dist2 = x_norm + c_norm - 2 * (X @ centers.T)      # (N,k)

        assign = dist2.argmin(dim=1)                       # (N,)

        # 割当が収束したら終了
        if prev_assign is not None and torch.equal(assign, prev_assign):
            break
        prev_assign = assign.clone()

        # 各クラスタの新しい中心を計算
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                centers[j] = X[mask].mean(dim=0)
            else:
                # 空クラスタはランダム再初期化
                ridx = torch.randint(0, N, (1,), device=X.device)
                centers[j] = X[ridx]

    return centers, assign


# =========================================
# 指定インデックス群の特徴量を一括抽出
# =========================================
def compute_features_for_indices(model, dataset, indices, device, batch_size=256, num_workers=2):
    """
    model.module.encoder を使用して特徴量を抽出
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    features_list = []
    idx_order = []

    with torch.no_grad():
        for bi, (imgs, *rest) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            feats = model.module.encoder(imgs)  # 特徴量抽出
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)  # Flatten
            features_list.append(feats.cpu())

            # 実際の元インデックスを記録
            start = bi * loader.batch_size
            bs = feats.size(0)
            idx_order.extend(indices[start:start+bs])

    features = torch.cat(features_list, dim=0) if len(features_list) > 0 else torch.empty(0)
    feat_dict = {idx: features[i] for i, idx in enumerate(idx_order)}
    return feat_dict


# =========================================
# メイン関数：k-meansを用いたバッファ更新
# =========================================
def set_replay_samples_kmeans(opt, model, prev_indices=None):
    """
    過去バッファの縮小はランダム（既存ロジックのまま）。
    新規に追加するサンプルはk-meansで代表点を選択。
    """
    is_training = model.training
    model.eval()

    # 軽量なTransform
    val_transform = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.ToTensor()
        ])

    # ===== データセット準備 =====
    if opt.dataset == 'cifar10':
        val_dataset = datasets.CIFAR10(root=opt.data_folder, transform=val_transform, download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'cifar100':
        val_dataset = datasets.CIFAR100(root=opt.data_folder, transform=val_transform, download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'tiny-imagenet':
        val_dataset = TinyImagenet(root=opt.data_folder, transform=val_transform, download=True)
        val_targets = val_dataset.targets
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    # ===== 既存バッファの扱い =====
    if prev_indices is None:
        prev_indices = []
        # 最初のタスクでは全てのクラスが対象
        observed_classes = list(range(0, opt.target_task * opt.cls_per_task))
    else:
        # バッファを過去タスク分に均等割り当てるため縮小（元ロジックそのまま）
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)
                keep = torch.tensor(_prev_indices)[mask]
                if keep.numel() > 0:
                    perm = torch.randperm(keep.numel())
                    prev_indices += keep[perm[:min(size_for_c, keep.numel())]].tolist()
            print(np.unique(val_targets[prev_indices], return_counts=True))

        # 今回新たに観測するクラス（直近タスク）
        observed_classes = list(range(max(opt.target_task - 1, 0) * opt.cls_per_task,
                                      opt.target_task * opt.cls_per_task))

    print("buffer_er.py observed_classes:", observed_classes)

    # 観測クラスがなければ終了
    if len(observed_classes) == 0:
        model.train(is_training)
        return prev_indices

    # ===== 観測クラスに属するインデックス =====
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)
    print("val_unique_cls:", val_unique_cls)
    print("opt.mem_size:", opt.mem_size)

    # ===== 特徴量を一括抽出 =====
    device = next(model.parameters()).device
    feat_cache = compute_features_for_indices(model, val_dataset, observed_indices, device=device)
    # print("feat_cache.shape: ", feat_cache.shape)

    # ===== k-meansによる代表点選択 =====
    selected_observed_indices = []

    for c_idx, c in enumerate(val_unique_cls):
        # クラス c の候補インデックス
        cls_mask = (val_targets[observed_indices] == c)
        cls_indices = (torch.tensor(observed_indices)[cls_mask]).tolist()

        # 残り枠に応じて配分数を計算（ceil/floorを確率pで制御）
        remain_cap = (opt.mem_size - len(prev_indices) - len(selected_observed_indices))
        remain_cls = (len(val_unique_cls) - c_idx)
        size_for_c_float = (remain_cap / remain_cls) if remain_cls > 0 else 0
        print(f"size_for_c_float (class={c}):", size_for_c_float)

        p = size_for_c_float - (remain_cap // remain_cls if remain_cls > 0 else 0)
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)

        if size_for_c <= 0 or len(cls_indices) == 0:
            continue
        size_for_c = min(size_for_c, len(cls_indices))

        # --- k-means で代表点を選ぶ ---
        feats = torch.stack([feat_cache[idx] for idx in cls_indices], dim=0)
        # 特徴を正規化して安定化（任意）
        feats = F.normalize(feats, p=2, dim=1)

        # k-means実行
        centers, assign = kmeans_torch(feats, k=size_for_c, max_iters=50, tol=1e-4)

        # 各クラスタから中心に最も近いサンプルを選択
        chosen = []
        for j in range(size_for_c):
            mask = (assign == j)
            if not mask.any():
                ridx = torch.randint(0, feats.size(0), (1,)).item()
                chosen.append(cls_indices[ridx])
            else:
                sub = feats[mask]
                ctr = centers[j].unsqueeze(0)
                dist2 = ((sub - ctr) ** 2).sum(dim=1)
                jmin = dist2.argmin().item()
                global_pos = torch.nonzero(mask, as_tuple=False).view(-1)[jmin].item()
                chosen.append(cls_indices[global_pos])

        # 念のため重複削除
        chosen = list(dict.fromkeys(chosen))
        if len(chosen) > size_for_c:
            chosen = chosen[:size_for_c]

        selected_observed_indices += chosen

    print(np.unique(val_targets[selected_observed_indices], return_counts=True))

    # 元のモードに戻す
    model.train(is_training)

    # 既存 + 新規
    return prev_indices + selected_observed_indices