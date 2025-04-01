import torch

class EarlyStopping:
    """ Early stops training if validation accuracy doesn't improve after a given patience. """
    
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:  # 🚀 Best Accuracy보다 낮으면 증가
            self.counter += 1
            if self.verbose:
                print(f"❌ EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True  # 🚀 Early Stopping 트리거
        else:
            self.best_score = score  # 🚀 Best Score 갱신
            self.counter = 0  # 🚀 카운터 리셋