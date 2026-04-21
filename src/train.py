import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import timm
import mlflow
import mlflow.pytorch
from dataset import get_dataloaders, CLASSES

def build_model(num_classes=4):
    return timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.batch_size, sample_fraction=args.sample_fraction)
    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    mlflow.set_experiment('brain-tumor-classification')
    with mlflow.start_run(run_name=f'efficientnet_b0_frac{args.sample_fraction}') as run:
        mlflow.log_params(vars(args))
        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
            vl, va = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            mlflow.log_metrics({'train_loss':tl,'train_acc':ta,'val_loss':vl,'val_acc':va}, step=epoch)
            print(f'Epoch {epoch:02d}/{args.epochs} | train_loss={tl:.4f} acc={ta:.3f} | val_loss={vl:.4f} acc={va:.3f}')
            if va > best_val_acc:
                best_val_acc = va
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'  Meilleur modele sauvegarde (val_acc={va:.3f})')
        mlflow.log_metric('best_val_acc', best_val_acc)
        mlflow.log_artifact('best_model.pth')
        mlflow.pytorch.log_model(model, name='model')
        print(f'Run ID: {run.info.run_id}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',       default='data')
    parser.add_argument('--epochs',          type=int,   default=5)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--sample_fraction', type=float, default=0.1)
    main(parser.parse_args())
