import os
import csv
import random
import numpy as np
import torch




def seed_everything(seed):
    # Python 内部のハッシュシードを固定（辞書等の再現性に寄与）
    # os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Python 標準の乱数生成器のシード固定
    random.seed(seed)
    
    # NumPy の乱数生成器のシード固定
    np.random.seed(seed)
    
    # PyTorch のシード固定
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # マルチGPU対応の場合
    # Deterministic モードの有効化（PyTorch の一部非決定的な処理の回避）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)



# モデルの保存
def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...'+save_file)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# csvファイルに値を書き込む
def write_csv(value, path, file_name, epoch):
    # ファイルパスを生成
    file_path = f"{path}/{file_name}.csv"

    # ファイルが存在しなければ新規作成、かつヘッダー行を記入する
    # value がリストの場合は、ヘッダーの値部分は要素数に合わせて "value_1", "value_2", ... とする例
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行を定義（必要に応じて適宜変更）
            if isinstance(value, list):
                header = ["epoch"] + [f"value_{i+1}" for i in range(len(value))]
            else:
                header = ["epoch", "value"]
            writer.writerow(header)

    # CSV に実際のデータを追加記録する
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if isinstance(value, list):
            row = [epoch] + value
        else:
            row = [epoch, value]
        writer.writerow(row)



# csvファイルに値を書き込む
def write_csv_analysis(value, path, file_name, task_model, task_data,):
    # ファイルパスを生成
    file_path = f"{path}/{file_name}.csv"

    # ファイルが存在しなければ新規作成、かつヘッダー行を記入する
    # value がリストの場合は、ヘッダーの値部分は要素数に合わせて "value_1", "value_2", ... とする例
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行を定義（必要に応じて適宜変更）
            if isinstance(value, list):
                header = ["task_model"] + ["task_data"] + [f"value_{i+1}" for i in range(len(value))]
            else:
                header = ["task_model", "task_data", "value"]
            writer.writerow(header)

    # CSV に実際のデータを追加記録する
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if isinstance(value, list):
            row = [task_model] + [task_data] + value
        else:
            row = [task_model, task_data, value]
        writer.writerow(row)