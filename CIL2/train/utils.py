

import torch




# =====================================================================
# Nearest Class Mean (NCM) 分類
# =====================================================================
def ncm_classify(val_encoded, val_labels, class_mean_encoded, metric="euclidean"):
    """
    Nearest Class Mean (NCM)分類を行う関数

    Parameters
    ----------
    val_encoded : torch.Tensor
        検証データの特徴量 [num_samples, feature_dim]
    val_labels : torch.Tensor
        検証データのラベル [num_samples]
    class_mean_encoded : dict
        各クラスの平均特徴量 {class_id: tensor([feature_dim])}
    metric : str, default="euclidean"
        距離指標の種類を指定
        - "euclidean" : ユークリッド距離
        - "cosine"    : コサイン類似度
        - "mahalanobis": マハラノビス距離 (共分散は全体のval_encodedから推定)

    Returns
    -------
    pred_labels : torch.Tensor
        予測ラベル [num_samples]
    accuracy : float
        分類精度 (0.0 ~ 1.0)
    """

    # クラス平均をまとめる
    class_ids = sorted(class_mean_encoded.keys())
    class_means = torch.stack([class_mean_encoded[c] for c in class_ids], dim=0)  # [num_classes, feature_dim]

    # =========================================
    # 距離計算
    # =========================================
    if metric == "euclidean":
        # ユークリッド距離
        # [N, C, D] → norm(dim=2) → [N, C]
        dists = torch.norm(val_encoded.unsqueeze(1) - class_means.unsqueeze(0), dim=2)

    elif metric == "cosine":
        # コサイン類似度 → 1 - cosine_similarityを距離として扱う
        # 正規化
        val_norm = torch.nn.functional.normalize(val_encoded, dim=1)      # [N, D]
        mean_norm = torch.nn.functional.normalize(class_means, dim=1)     # [C, D]
        # cos_sim: [N, C]
        cos_sim = torch.matmul(val_norm, mean_norm.T)
        dists = 1.0 - cos_sim  # 類似度が高いほど距離が小さくなる

    elif metric == "mahalanobis":
        # マハラノビス距離
        # 共分散行列をval_encoded全体から推定
        X = val_encoded - val_encoded.mean(dim=0)
        cov = torch.matmul(X.T, X) / (X.shape[0] - 1)  # [D, D]
        # 数値安定性のための微小項を加えて逆行列を計算
        cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0]))

        # 各サンプルと各クラス平均間のマハラノビス距離を計算
        N, D = val_encoded.shape
        C = class_means.shape[0]
        dists = torch.zeros((N, C))
        for i in range(C):
            diff = val_encoded - class_means[i]  # [N, D]
            # (x - μ)^T Σ^-1 (x - μ)
            dists[:, i] = torch.sqrt(torch.sum((diff @ cov_inv) * diff, dim=1))

    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'euclidean', 'cosine', 'mahalanobis'.")

    # =========================================
    # 予測クラスを決定
    # =========================================
    pred_indices = torch.argmin(dists, dim=1)  # [N]
    pred_labels = torch.tensor([class_ids[i] for i in pred_indices], dtype=torch.long)

    # 精度計算
    correct = (pred_labels == val_labels).sum().item()
    accuracy = correct / len(val_labels)

    print(f"[{metric}] Nearest Class Mean 分類精度: {accuracy:.4f} ({correct}/{len(val_labels)})")

    return pred_labels, accuracy