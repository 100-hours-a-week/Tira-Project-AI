import os
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init

class AverageMeter:
    """Computes and stores the average, current value, and count"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        """Update meter with new value"""
        if isinstance(val, torch.Tensor):  # ✅ val이 Tensor이면 item()으로 변환
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n
        
        if self.count > 0:  # ✅ ZeroDivision 방지
            self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum()
        res.append(correct_k.mul(100.0 / batch_size))  # ✅ 메모리 안정성 향상 (mul_() → mul())
    return res


def visualize_pca(X, y, title="pca Visualization"):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"class {label}", alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.show()

def visualize_tsne(X, y, title="t-SNE Visualization"):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8,6))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"class {label}", alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.show()