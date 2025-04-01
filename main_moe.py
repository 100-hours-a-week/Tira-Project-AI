import argparse
import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import os
from tqdm import tqdm
from datetime import datetime
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from dataloader.my_dataloader_moe import get_dataloader
# from model import model_dict
from model import SharedEncoder, SwitchGate, SwitchMoE, MoEWithSharedEncoder
from utils import AverageMeter, accuracy, visualize_pca, visualize_tsne
from earlystopping import EarlyStopping
 
from torchsummary import summary
import wandb

import datetime

import glob



parser = argparse.ArgumentParser()
parser.add_argument("--shared_model", type=str, default="resnet50")
# parser.add_argument("--root", type=str, default="../data/")
parser.add_argument("--num_experts", type=int, default=3)
parser.add_argument("--classes_num", type=int, default=100)
parser.add_argument("--dataset",type=str,default="4",
    choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],help="dataset")
# parser.add_argument("--num_tokens", type=int, default=1000)


parser.add_argument("--shared_hidden_dim", type=int, default=512)
parser.add_argument("--expert_input_dim", type=int, default=1024)
parser.add_argument("--epsilon", default=0.03, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--patience", type=int, default=2)

# parser.add_argument("--classes_num", type=int, default=100)


# parser.add_argument("--momentum", type=float, default=0.9)
# parser.add_argument("--weight-decay", type=float, default=5e-4)
# parser.add_argument("--gamma", type=float, default=0.1)
# parser.add_argument("--milestones", type=int, nargs="+", default=[150, 180, 210])
# parser.add_argument("--seed", type=int, default=1)
# parser.add_argument("--gpu-id", type=int, default=0)

parser.add_argument("--print_freq", type=int, default=5)
parser.add_argument("--aug_nums", type=int, default=2)
# parser.add_argument("--exp_postfix", type=str, default="run1")


# parser.add_argument("--T", type=float)
# parser.add_argument("--alpha", type=float)


args = parser.parse_args()

timestamp = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H%M%S")

run_name = "{}/{}/{}".format(args.dataset, args.shared_model, timestamp)
wandb.init(project="ktb_moe", name=run_name, config={
    "learning_rate": args.lr,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "model": args.shared_model,
})


# # current model & version folder create
# exp_name = "_".join(args.shared_model) + args.exp_postfix
# exp_path = "./moe/{}/{}".format(args.dataset, exp_name)
# os.makedirs(exp_path, exist_ok=True)
# print(exp_path)



def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()
    
    scaler = torch.cuda.amp.GradScaler() 

    # pbar = tqdm(train_loader, total=len(train_loader), desc="Training", leave=True)
    # for i, data in enumerate(pbar):  
    for i, data in enumerate(train_loader):       
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)

        optimizer.zero_grad()  

        with torch.cuda.amp.autocast():
            out = model(imgs) # no aux loss
            # out, _ = model(imgs) # no aux loss
            # out, aux_loss = model(imgs) # aux loss
            loss = F.cross_entropy(out, label)
            # loss += lambda_aux * aux_loss
        scaler.scale(loss).backward()  # amp applied loss
        scaler.step(optimizer)  
        scaler.update()  

        loss_recorder.update(loss.item(), n=imgs.size(0))
        acc = accuracy(out, label)[0]
        acc_recorder.update(acc.item(), n=imgs.size(0))

    # # batch별 tqdm 출력
    # pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc.item():.2f}%"})
        

    losses = loss_recorder.avg
    acces = acc_recorder.avg
    return losses, acces



def evaluation(model, val_loader):
    model.eval()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()

    with torch.no_grad():
        for img, label in val_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out= model(img)
            # out, _ = model(img)
            acc = accuracy(out, label)[0]
            loss = F.cross_entropy(out, label)
            acc_recorder.update(acc.item(), img.size(0))
            loss_recorder.update(loss.item(), img.size(0))

    losses = loss_recorder.avg
    acces = acc_recorder.avg
    return losses, acces


def train(model, optimizer, train_loader, val_loader, device, patience):
    best_acc = -1
    early_stopping = EarlyStopping(patience=patience)
    # f = open(os.path.join(exp_path, "log_test.txt"), "w") # traing logging !!!
    
    for epoch in tqdm(range(args.epochs)):
        train_losses, train_acces = train_one_epoch(model, optimizer, train_loader, device)
        val_losses, val_acces = evaluation(model, val_loader)

        # W&B 로깅
        wandb.log({
            "epoch": epoch,
            "train_loss": train_losses,
            "train_accuracy": train_acces,
            "val_loss": val_losses,
            "val_accuracy": val_acces,
            "gate1_acc": gate1_acc
        })

        if val_acces > best_acc:
            best_acc = val_acces
            # state_dict = dict(epoch = epoch+1, model=model.state_dict(), acc=val_acces)
            # name = os.path.join(exp_path, args.model_name, "ckpt", "best.pth")
            # torch.save(state_dict, name)

        # scheduler.step()


        print(f"Epoch {epoch+1}: Train Loss {train_losses:.4f} | Train Acc {train_acces:.2f}% || "
              f"Val Loss {val_losses:.4f} | Val Acc {val_acces:.2f}%")


        early_stopping(val_acces)
        if early_stopping.early_stop:
            print("Early stopping activated. Training stopped.")
            break 
    print(" Training finished! model:{} best acc:{:.2f}".format(args.shared_model, best_acc))
    wandb.finish()



if __name__ == "__main__":
    # train_loader, test_loader, val_loader = get_dataloader(args)
    train_loader, _, val_loader = get_dataloader(args)


    # # V1 
    # encoder = SharedEncoder(args.shared_model, output_dim=args.hidden_dim, num_class=args.classes_num)
    
    # V2
    moe_model = MoEWithSharedEncoder(
        shared_model = args.shared_model,
        shared_hidden_dim = args.shared_hidden_dim,
        expert_input_dim = args.expert_input_dim,
        num_experts = args.num_experts,
        num_class=args.classes_num
    )

    # experts name 확인용
    for name, param in moe_model.shared_encoder.named_parameters():
        # print(f"name: {name}, param: {param}")
        print(f"name: {name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = moe_model.to(device)


    summary(model, input_size=(3, 224, 224))

    optimizer = optim.SGD(
        model.parameters(),
        lr = args.lr
    )    


    # scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
    train(model, optimizer, train_loader, val_loader, device, patience=args.patience)

