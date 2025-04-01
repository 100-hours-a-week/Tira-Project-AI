import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os 
from collections import Counter
import matplotlib.pyplot as plt

from kmeans_undersampling import kmeans_undersample


def get_dataloader(args):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(p=0.5),  # 랜덤 좌우 반전
        transforms.RandomRotation(10),  # 랜덤 회전 (-10도 ~ +10도)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변형
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 랜덤 이동
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ✅ 2. 사용할 데이터셋 목록 정의
    dataset_paths = {
        # "3+4+7": "./datasets/3_4_7"
        # "3": "./datasets/3. fruit-and-vegetable-image-recognition",
        # "4": "./datasets/4. vegetable-image-dataset",
        # "5": "./datasets/5. flowers-dataset",
        # "7": "./datasets/7. tomato-disease-multiple-sources",
        "9": "./datasets/9. orange-disease-dataset"
    }
    train_loaders = []
    val_loaders = []
    # test_loaders = []
    num_workers = min(8, os.cpu_count())

    for dataset_name, base_path in dataset_paths.items():
        train_dir = os.path.join(base_path, "train")
        val_dir = os.path.join(base_path, "validation")
        test_dir = os.path.join(base_path, "test")

        if os.path.exists(train_dir):
            train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
            train_labels = [label for _, label in train_dataset]

        if os.path.exists(val_dir):
            val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
            val_labels = [label for _, label in val_dataset]
    

    return train_loaders, None, val_loaders