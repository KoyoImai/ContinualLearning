

import torch
import torch.nn.functional as F


# =====================================================
# Experience Feature Matrix (EFM) を計算
# Projector の出力変化で outputs がどう変化するかを計算
# =====================================================
def postprocess_prco(opt, model, train_loader, feat=True):

    # 変数初期化
    E_sum = None
    batch_count = 0

    # model を検証モードに変更
    model.eval()


    for idx, (images, labels) in enumerate(train_loader):
    # for idx, data in enumerate(train_loader):
    #     images, labels = data

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # print("images.shape: ", images.shape)    # images.shape:  torch.Size([512, 3, 32, 32])
        # print("labels.shape: ", labels.shape)    # labels.shape:  torch.Size([512])

        # 特徴量獲得
        # with torch.no_grad():  # with torch.no_grad()は，2025/10/13に追加
        encoded, features, output, _, _, _ = model(images)
        # print("features.shape: ", features.shape)   # features.shape:  torch.Size([512, 128])
        # print("output.shape: ", output.shape)       # output.shape:  torch.Size([512, 10])

        # outputの形状を復元
        logits = output

        # 温度スケーリング
        # tau = 0.1
        tau = opt.temp_prco_efm
        logits = logits / tau

        # log-softmax と確率
        logp = F.log_softmax(logits, dim=1)         # (B, C)
        p    = logp.exp()                           # (B, C)

        # print("logp.shape: ", logp.shape)       # logp.shape:  torch.Size([512, 10])
        # print("p.shape: ", p.shape)             # p.shape:  torch.Size([512, 10])

        assert features.requires_grad is True

        # バッチサイズB，クラス数C
        B, C = logp.shape

        # projectorかEncoder出力次元数
        if feat:
            D = features.shape[1]          # 特徴次元 (= 128)
        else:
            D = encoded.shape[1]

        # どのクラスで期待値を取るか
        topk = None

        if topk is None or topk >= C:
            idx_sel = torch.arange(C, device=logp.device).unsqueeze(0).expand(B, C)   # (B, C)
        else:
            _, idx_sel = torch.topk(p, k=topk, dim=1)  # (B, K)


        # 各サンプルの局所行列を格納
        E_local_list = []


        #平均（期待値）を取る前に，1サンプルごとに勾配を計算
        for b in range(B):

            # print("b: ", b)

            # 1) クラスごとの ∂ log p_c / ∂ f_b を取る
            grads = []
            for c in idx_sel[b].tolist():
                
                # logp[b, c] はスカラー。features 全体 (B, D) への勾配を取り、当該サンプル b の行だけ抜く
                if feat:
                    g_bc_full = torch.autograd.grad(logp[b, c], features, retain_graph=True, create_graph=False)[0]  # (B, D)
                else:
                    g_bc_full = torch.autograd.grad(logp[b, c], encoded, retain_graph=True, create_graph=False)[0]  # (B, D)

                g_bc = g_bc_full[b]   # (D,)
                # print("g_bc: ", g_bc)
                # print("g_bc.shape: ", g_bc.shape) # g_bc.shape:  torch.Size([128])
                grads.append(g_bc)

            G = torch.stack(grads, dim=0)    # (K, D)  ※ K = C もしくは topk
            # print("G.shape: ", G.shape)    # G.shape:  torch.Size([10, 128])

            # 2) 期待値の重み（確率）を用意
            weights = p[b, idx_sel[b]]                    # (K,)
            w = weights / (weights.sum() + 1e-12)         # 念のため正規化

            # 3) E_f(x_b) = Σ_c w_c g_c g_c^T = (G^T * w) @ G
            #    （各行に重みを掛けてから二次形式）
            E_b = (G.T * w) @ G                           # (D, D)

            E_local_list.append(E_b)    

        # 4) バッチの局所行列テンソル
        E_local = torch.stack(E_local_list, dim=0)        # (B, D, D)
        # print("E_local.shape: ", E_local.shape)         # E_local.shape:  torch.Size([512, 128, 128])

        # 5) バッチ平均（このバッチの E を作る段階。全データ平均は次ステップで）
        E_batch = E_local.mean(dim=0).detach().cpu()      # (D, D)

        
        if E_sum is None:
            E_sum = E_batch.clone()
        else:
            E_sum = E_sum + E_batch.cpu()
        # E_sum = E_batch if E_sum is None else (E_sum + E_batch)
        batch_count += 1

        # --- デバッグ ---
        if idx == 0:
            print("E_local:", E_local.shape, "E_batch:", E_batch.shape)
            # 例: E_local (B, 128, 128), E_batch (128, 128)

        # デバッグ用
        # if batch_count > 1:
        #     break
    
    print("batch_count: ", batch_count)


    # ループ終了後：
    E = E_sum / batch_count                     # (D, D) 全バッチ平均の EFM

    # 数値誤差で非対称成分が出ないよう軽く対称化（PSD を保つ目的）
    E = 0.5 * (E + E.T)
    print("E.shape: ", E.shape)


    # 固有分解（半正定値なので e i g h が安全）
    eigvals, eigvecs = torch.linalg.eigh(E)     # eigvals: (D,), 昇順; eigvecs: (D, D) 列が固有ベクトル

    # 重要方向＝固有値の大きい順に並べ替え
    idx_desc = torch.argsort(eigvals, descending=True)
    lam = eigvals[idx_desc]                     # (D,)
    U   = eigvecs[:, idx_desc]                  # (D, D) 各列が重要度順の固有ベクトル
    # print("lam.shape: ", lam.shape)    # lam.shape:  torch.Size([128])
    # print("U.shape: ", U.shape)        # U.shape:  torch.Size([128, 128])

    # タスク終了時の Experience Feature Matrix (EFM)
    model.module.efm = E
    model.module.lam = lam
    model.module.U = U


    # model を訓練モードに変更
    model.train()

    
    return None




