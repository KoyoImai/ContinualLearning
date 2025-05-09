

import torch
from collections import defaultdict

from models.utils_hooks import FeatureHook, register_resnet18_hooks



def extract_features_cclis(opt, model, data_loader):


    # modelをevalモードに変更
    model.eval()

    # 特徴量とラベルを保存するリスト
    features_list = []
    labels_list = []

    # Hookの設定
    granularity = getattr(opt, "granularity", opt.block_type)  # ← optに属性があれば使う（なければ"block"）
    hooker = register_resnet18_hooks(model, opt, granularity=granularity)

    # 特徴マップの整形方法
    feat_mode = getattr(opt, "feature_reduce_mode", opt.flatten_type)  # デフォルト: avgpool

    # 各層の特徴を保存する辞書（名前ごとにリスト）
    layerwise_features = defaultdict(list)

    with torch.no_grad():

        # 特徴とラベルの抽出
        for (images, labels) in data_loader:
            
            # print("images.shape: ", images.shape)
            # print("labels.shape: ", labels.shape)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            if opt.use_dp:
                feature = model.module.encoder(images)
            else:
                feature = model.encoder(images)
            # print("feature.shape: ", feature.shape)

            # 各層のhook出力（平均プーリングしてflatten）
            for name, feat in hooker.outputs.items():
                print("feat.shape: ", feat.shape)
                if feat_mode == "avgpool":
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
                elif feat_mode == "flatten":
                    feat = torch.nn.functional.interpolate(feat, scale_factor=0.5,
                                                           mode="bilinear", align_corners=False)
                    feat = feat.view(feat.size(0), -1)  # [B, C×H×W]
                else:
                    raise ValueError(f"Unknown feature_reduce_mode: {feat_mode}")
                
                print("feat.shape: ", feat.shape)

                layerwise_features[name].append(feat)


            # listに保存
            features_list.append(feature)
            labels_list.append(labels)


    # listをtorch.tensorに変換
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    # print("features.sahpe: ", features.shape)
    # print("labels.shape: ", labels.shape)

    layer_outputs = {name: torch.cat(feats, dim=0) for name, feats in layerwise_features.items()}
        
    return features, labels, layer_outputs





def extract_features_cclis_projector(opt, model, data_loader):


    # modelをevalモードに変更
    model.eval()

    # 特徴量とラベルを保存するリスト
    features_list = []
    labels_list = []

    
    with torch.no_grad():

        # 特徴とラベルの抽出
        for (images, labels) in data_loader:
            
            # print("images.shape: ", images.shape)
            # print("labels.shape: ", labels.shape)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            feature, encoded = model(images, return_feat=True)
            # print("feature.shape: ", feature.shape)

            # listに保存
            features_list.append(feature)
            labels_list.append(labels)


    # listをtorch.tensorに変換
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    # print("features.sahpe: ", features.shape)
    # print("labels.shape: ", labels.shape)
        
    return features, labels