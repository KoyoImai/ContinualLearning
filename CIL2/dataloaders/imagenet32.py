
import os
import pickle
import numpy as np
from PIL import Image

import torch



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_data(input_file):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))

    return x, y


class ImageNet32(torch.utils.data.Dataset):

    def __init__(self, opt, transform, root):

        super().__init__()

        # データセットまでのパス
        file_path = f"{root}/Imagenet32/train/train_data_batch_{opt.target_task+1}"
        self.file_path = file_path

        # file_pathの読み取り
        d = unpickle(self.file_path)
        x = d['data']
        y = d['labels']

        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3))

        # train_data_batch_1　の場合の形状確認
        # print("x.shape: ", x.shape)             # x.shape:  (128116, 32, 32, 3)
        # print("len(y): ", len(y))               # len(y):  128116
        # print("len(set(y)): ", len(set(y)))     # len(set(y)):  1000


        self.images = x  # shape: (N, 32, 32, 3), dtype=uint8
        self.labels = y  # shape: (N,), dtype=int64
        self.transform = transform


    def __getitem__(self, idx):

        img = self.images[idx]
        img = Image.fromarray(img)

        data = self.transform(img)


        label = torch.as_tensor(self.labels[idx], dtype=torch.long)

        return data, label



    def __len__(self):
        """サンプル数（= 読み込んだ train_data_batch_n 内の件数）"""
        return self.images.shape[0]