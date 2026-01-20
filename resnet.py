import os
import pandas as pd
import numpy as np
import cv2 as cv
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate, Dataset
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from datetime import datetime
from colorama import Fore, Style, init


# 导入自定义模块 
from std_resnet import resnet18
from metric import accuracy
from seed import set_seed
from Draw import plot_confusion
except ImportError:
    plot_confusion = None 

# 初始化
init()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


CFG = {
    'batch_size': 10,
    "num_workers": 0, 
    'pin_memory': False,
    "use_mixup_or_cutmix": True, 
    'amp': True,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "warmup_epochs": 0,
    "patience": 0,
    "label_smoothing": 0.1,
    "weight_decay": 1e-5,
    "epochs": 200,
    "lr_range": [2e-4 * i for i in range(1, 10, 2)], 
    "seed": 42
}


# (Data Preparation)
def load_data(csv_path="./classification_image/MetaData.csv", img_root="./classification_image/"):
    """
    读取CSV和图像数据
    注意：Github仓库中请提供示例MetaData.csv和对应的文件夹结构
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Metadata file not found at {csv_path}. Please check directory structure.")
        return [], [], {}, []

    info = pd.read_csv(csv_path).values
    file_names = info[:, 0]
    label_names = info[:, 1]
    
    label_set = sorted(list(set(label_names)))
    name2label = {n: i for i, n in enumerate(label_set)}
    labels = np.array([name2label[name] for name in label_names])
    
    data = []
    print("Loading images...")
    for f in file_names:
        img_path = os.path.join(img_root, f)
        img = cv.imread(img_path)
        if img is None:
            print(f"Error: Cannot read image {img_path}")
            continue
        data.append(img)
    
    return data, labels, name2label, label_set


# (Augmentation)
# 计算的均值和方差 
pixels_mean = [0.485, 0.456, 0.406] 
pixels_std = [0.229, 0.224, 0.225]

Crop_Size = 200

class RandAugmentWrapper(A.ImageOnlyTransform):
    def __init__(self, num_ops=2, magnitude=9, always_apply=False, p=0.5):
        super(RandAugmentWrapper, self).__init__(always_apply, p)
        self.rand_augment = v2.RandAugment(num_ops=num_ops, magnitude=magnitude)
        self.num_ops = num_ops
        self.magnitude = magnitude

    def apply(self, image, **params):
        # Input image is numpy array, RandAugment expects PIL or Tensor
        # v2.RandAugment can handle tensor, so we might need conversion if using older versions
        # Here assuming image is HWC numpy array
        return self.rand_augment(torch.from_numpy(image).permute(2,0,1)).permute(1,2,0).numpy()

    def get_transform_init_args_names(self):
        return ("num_ops", "magnitude")

class CutMixOrMixup:
    def __init__(self, n_class):
        self.cutmix = v2.CutMix(num_classes=n_class, alpha=1.0)   
        self.mixup  = v2.MixUp(num_classes=n_class, alpha=0.2)   
        self.cutmix_or_mixup = v2.RandomChoice([self.cutmix, self.mixup], p=[0.5, 0.5])
        
    def collate_fn(self, batch):
        images, labels = default_collate(batch)   
        return self.cutmix_or_mixup(images, labels)

train_tf = A.Compose([
    A.Resize(224, 224),
    A.RandomResizedCrop(size=(Crop_Size, Crop_Size), scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    A.HorizontalFlip(p=0.5),
    RandAugmentWrapper(num_ops=1, magnitude=9, p=1.0), 
    A.Normalize(mean=pixels_mean, std=pixels_std),
    ToTensorV2()               
])

test_tf = A.Compose([
    A.Resize(224, 224),
    A.CenterCrop(height=Crop_Size, width=Crop_Size),
    A.Normalize(mean=pixels_mean, std=pixels_std),
    ToTensorV2() 
])

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=img)["image"]
        else:
            augmented = torch.tensor(img, dtype=torch.float32)
            
        return augmented, torch.tensor(label, dtype=torch.long)


#  (Training Loop)

def train_one_epoch(epoch, model, optimizer, criterion, scaler, train_loader, device):
    model.train()
    running_loss, running_top1 = 0., 0.
    pbar = tqdm(train_loader, ncols=100, desc=f'Epoch {epoch}')
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if CFG['amp'] and scaler is not None:
            with autocast(dtype=torch.float16): # device_type='cuda' implied
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
        # Accuracy calculation
        top = accuracy(logits, labels, topk=(1,), train=True if CFG["use_mixup_or_cutmix"] else False)
        top1 = top[0]
        running_loss += loss.item() * imgs.size(0)
        running_top1 += top1 * imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}", top1=f"{top1:.2f}%")
        
    return running_loss / len(train_loader.dataset), running_top1 / len(train_loader.dataset)

def validate(model, criterion, test_loader, device):
    model.eval()
    with torch.no_grad():
        running_loss, running_top1 = 0., 0.
        for imgs, labels in tqdm(test_loader, ncols=100, desc='Val'):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            top = accuracy(logits, labels, topk=(1,), train=False)
            top1 = top[0]
            running_loss += loss.item() * imgs.size(0)
            running_top1 += top1 * imgs.size(0)

    return running_loss / len(test_loader.dataset), running_top1 / len(test_loader.dataset)

def main():
   
    set_seed(CFG['seed'])
    seed_worker = set_seed(CFG['seed'])

  
    data, labels, name2label, label_set = load_data()
    if len(data) == 0:
        print("No data loaded. Exiting.")
        return

    CFG["NUM_CLASSES"] = len(name2label)
    print(f"Classes: {name2label}")

   
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, shuffle=True, 
        random_state=2021, stratify=labels
    )
    
    
    metrics = []
    for lr in CFG['lr_range']:
        print(f"\nTraining with Learning Rate: {lr:.2e}")
        
        # Dataset & DataLoader
        cm = CutMixOrMixup(n_class=CFG["NUM_CLASSES"])
        train_data = CustomDataset(X_train, y_train, transform=train_tf)
        test_data = CustomDataset(X_test, y_test, transform=test_tf)
        
        train_loader = DataLoader(
            train_data, batch_size=CFG['batch_size'], shuffle=True,
            pin_memory=CFG['pin_memory'],
            collate_fn=cm.collate_fn if CFG["use_mixup_or_cutmix"] else None,
            worker_init_fn=seed_worker
        )
        test_loader = DataLoader(
            test_data, batch_size=CFG['batch_size'], shuffle=False,
            pin_memory=CFG['pin_memory']
        )

        # Model Init (Kaiming Init inside std_resnet)
        model = resnet18(num_classes=CFG["NUM_CLASSES"]).to(CFG['device'])
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=CFG['weight_decay'])
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
        scaler = GradScaler() if CFG['amp'] else None
        
        # Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

        # Training Loop
        best_acc = 0.0
        log_dir = f"log_pytorch/ResNet18_{lr}_{datetime.now():%m%d_%H%M}"
        os.makedirs(log_dir, exist_ok=True)
        
        for epoch in range(CFG['epochs']):
            train_loss, train_acc = train_one_epoch(epoch, model, optimizer, criterion, scaler, train_loader, CFG['device'])
            val_loss, val_acc = validate(model, criterion, test_loader, CFG['device'])
            scheduler.step()

            print(f"Epoch {epoch}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(log_dir, "model_best.pth"))
        
        # Final Evaluation
        print(f"Best Accuracy for lr={lr}: {best_acc:.2f}%")
        
        # Load best model for metrics
        model.load_state_dict(torch.load(os.path.join(log_dir, "model_best.pth")))
        model.eval()
        
        # Predict
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(CFG['device'])
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())
        
        # Calculate Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        metrics.append([lr, acc, f1])
        
        if plot_confusion:
            cm_matrix = confusion_matrix(all_labels, all_preds)
            plot_confusion(cm_matrix, f"ResNet18_lr{lr}", name2label, lr)
            
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Save Results
    pd.DataFrame(metrics, columns=["LR", "Accuracy", "F1"]).to_csv("training_results.csv", index=False)
    print("Training Completed.")

if __name__ == '__main__':
    main()
