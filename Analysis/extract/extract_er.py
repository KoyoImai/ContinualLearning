

import torch



def extract_features_er(opt, model, data_loader):


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
            
            y_pred, feature = model(x=images, return_feat=True)
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