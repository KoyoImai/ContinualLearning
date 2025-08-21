


# -----------------------------
# 2D 可視化メイン
# -----------------------------
def plot_loss_landscape_2d(
    model, criterion, loader,
    x_range=(-1,1,51), y_range=(-1,1,51),
    dir_type='random',
    base_ckpt_path=None,
    second_ckpt_path=None,
    skip_bn_and_bias=True,
    filter_normalize=True,
    rng=123,
    max_batches=None,          # 1点あたり何バッチ平均するか（None=全バッチ）
    save_png='landscape_2d.png'
    ):
    

    assert False