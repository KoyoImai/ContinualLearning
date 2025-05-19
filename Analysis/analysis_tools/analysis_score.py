
import os
import torch
import numpy as np
# from bhtsne import tsne
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score, mutual_info_score, homogeneity_score, completeness_score



def extract_features(model, target, inputs):
    feature = None
    
    def forward_hook(module, inputs, outputs):
        global features
        features = outputs.detach().clone()
        
    handle = target.register_forward_hook(forward_hook)
    model.eval()
    model(inputs)
    
    handle.remove()
    return features



class Eval_cluster:
    def __init__(self, model, dataloader, target_module, num_features):
        self.model = model
        self.dataloader = dataloader
        self.target_module = target_module
        self.num_features = num_features
    
    def _cal_disp(self, features, labels):

        # クラス内分散・クラス間分散を計算して返す

        n_samples, n_features = features.shape
        unique_labels = np.unique(labels)
        k = len(unique_labels)

        mean_all = np.mean(features, axis=0)  # 全体平均
        tr_W = 0.0  # クラス内分散（スカラー）
        tr_B = 0.0  # クラス間分散（スカラー）

        for label in unique_labels:
            X_q = features[labels == label]
            n_q = X_q.shape[0]
            mean_q = np.mean(X_q, axis=0)

            # クラス内分散: sum ||x - c_q||^2
            tr_W += np.sum((X_q - mean_q) ** 2)

            # クラス間分散: n_q * ||c_q - c_E||^2
            tr_B += n_q * np.sum((mean_q - mean_all) ** 2)

        return tr_W, tr_B
        
    def _cal_score(self, features, labels):
        """
        Calinski and Harabasz index: T. Calinski and J. Harabasz, 1974.
        
        s = \frac{tr(B_{k})}{tr(W_{k})}\times \frac{n_{E} - k}{k - 1}
        Inter-class variance: W_{k} = \sum_{q=1}^{k}\sum_{x\in C_{q}} (x - c_{q})(x - c_{q})^{\top}
        Intra-class variance: B_{k} = \sum_{q=1}^{k}n_{q}(c_{q} - c_{E})(c_{q} - c_{E})^{\top}
        """
        cal_score = calinski_harabasz_score(features, labels)
        return cal_score
    
    def _sil_score(self, features, labels):
        """
        Silhouette coefficient: P. J. Rousseeuw, 1987.
        
        s^{(i)} = \frac{b^{(i)} - a^{(i)}}{\max(a^{(i)}, b^{(i)})}
        Compactness of intra-class: a^{(i)} = \frac{1}{|C_{in}| - 1}\sum_{x^{(j)}\in C_{in}}\left\|x^{(i)} - x^{(j)}\right\|
        Inter-class variance: b^{(i)} = \frac{1}{|C_{near}| - 1}\sum_{x^{(j)}\in C_{near}}\left\|x^{(i)} - x^{(j)}\right\|
        """
        sil_score = silhouette_score(features, labels)
        return sil_score
    
    def _homo_score(self, ground_truth, predicted):
        """
        Homogeneity score
        
        h =
          \begin{cases}
            1 & \mathrm{if~} H(C,K) = 1\\
            1 - \frac{H(C|K)}{H(C)} & \mathrm{else}
          \end{cases}
        H(C|K) = -\sum_{k=1}^{|K|}\sum_{c=1}^{|C|}\frac{a_{c,k}}{N}\log \frac{a_{c,k}}{\sum_{c=1}^{|C|}a_{c,k}}
        H(C) = -\sum_{c=1}^{|C|}\frac{\sum_{k=1}^{|K|}a_{c,k}}{n}\log \frac{\sum_{k=1}^{|K|}a_{c,k}}{n}
        """
        homo_score = homogeneity_score(ground_truth, predicted)
        return homo_score
    
    def _comp_score(self, ground_truth, predicted):
        """
        Completeness score
        
        c =
          \begin{cases}
            1 & \mathrm{if~} H(K,C) = 0\\
            1 - \frac{H(K|C)}{H(K)} & \mathrm{else}
          \end{cases}
        H(K|C) = -\sum_{c=1}^{|C|}\sum_{k=1}^{|K|}\frac{a_{c,k}}{N}\log \frac{a_{c,k}}{\sum_{k=1}^{|K|}a_{c,k}}
        H(K) = -\sum_{k=1}^{|K|}\frac{\sum_{c=1}^{|C|}a_{c,k}}{n}\log \frac{\sum_{c=1}^{|C|}a_{c,k}}{n}
        """
        comp_score = completeness_score(ground_truth, predicted)
        return comp_score
        
    def computing_score(self):
        predicted_labels = torch.zeros(len(self.dataloader.dataset))
        ground_truth = torch.zeros(len(self.dataloader.dataset))
        low_level_features = torch.zeros(len(self.dataloader.dataset), self.num_features).to(torch.float64)
        
        cnt = 0
        for idx, (inputs, target) in enumerate(self.dataloader):
            inputs, target = inputs.cuda(), target.cuda()
            batch = inputs.size(0)
            
            with torch.no_grad():
                pred = self.model(inputs).softmax(dim=1).argmax(dim=1).data.cpu()
            predicted_labels[cnt:cnt+batch] = pred
            ground_truth[cnt:cnt+batch] = target.data.cpu()
            
            features = extract_features(self.model, self.target_module, inputs).view(batch, -1)
            low_level_features[cnt:cnt+batch] = features.data.cpu()
            cnt += batch
        
        sil_score = self._sil_score(low_level_features.numpy(), ground_truth.numpy())
        cal_score = self._cal_score(low_level_features.numpy(), ground_truth.numpy())
        homo_score = self._homo_score(ground_truth.numpy(), predicted_labels.numpy())
        comp_score = self._comp_score(ground_truth.numpy(), predicted_labels.numpy())
        
        print('Sil. score: %.6f' % sil_score)
        print('Cal. score: %.6f' % cal_score)
        print('Homo score: %.6f' % homo_score)
        print('Comp score: %.6f' % comp_score)
        
    def visualize_low_level_features(self, save_name='figures', random_seed=-1, plot_size=0.5):
        os.makedirs(save_name, exist_ok=True)
        ground_truth = torch.zeros(len(self.dataloader.dataset))
        low_level_features = torch.zeros(len(self.dataloader.dataset), self.num_features).to(torch.float64)
        
        cnt = 0
        for idx, (inputs, target) in enumerate(self.dataloader):
            inputs, target = inputs.cuda(), target.cuda()
            batch = inputs.size(0)
            
            ground_truth[cnt:cnt+batch] = target.data.cpu()
            features = extract_features(self.model, self.target_module, inputs).view(batch, -1)
            low_level_features[cnt:cnt+batch] = features.data.cpu().to(torch.float64)
            cnt += batch
        
        X = tsne(low_level_features.numpy(), dimensions=2, perplexity=30, rand_seed=random_seed)
        xcoords, ycoords = X[:,0], X[:,1]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colorlist = [colors[int(index.item())] for index in ground_truth]
        
        plt.scatter(xcoords, ycoords, color=colorlist, s=plot_size)
        plt.savefig((os.path.join(save_name, 't-SNE-scatter.png')))
        plt.savefig((os.path.join(save_name, 't-SNE-scatter.pdf')))