import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import mlflow
import timm
from dataset import get_dataloaders, CLASSES

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_dataloaders('data', batch_size=32, sample_fraction=0.1)
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=3)
    print(report)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predit'); ax.set_ylabel('Reel')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print('Matrice sauvegardee -> confusion_matrix.png')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'F1-macro = {f1:.3f}')

if __name__ == '__main__':
    main()
