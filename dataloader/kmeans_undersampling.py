import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import torch

# img -> feat vec.
# 시각화 또는 KMeans undersampling - 학습에는 넣지 않는게 좋음. 빠르게 학습 끝내야지 성능 영향 없
def extract_features(dataset, model):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for img, label in dataset:
            if isinstance(img, torch.Tensor):
                img = img.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            else:
                continue
            feat = model(img).squeeze().flatten().numpy()
            features.append(feat)
            labels.append(label)
    return np.array(features), np.array(labels) # X, y


def kmeans_undersample(X, y):
    X_new, y_new = [], []
    min_count = min(Counter(y).values())

    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        X_class = X[idxs]

        kmeans = KMeans(n_clusters=min_count, random_state=42)
        kmeans.fit(X_class)
        X_new.extend(kmeans.cluster_centers_)
        y_new.extend([label] * min_count)

    return np.array(X_new), np.array(y_new)


def make_loader_from_features(X, y, batch_size=64):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)