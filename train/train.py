import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os
from datetime import datetime

# FIX SEED
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# CREATE UNIQUE RUN FOLDER
base_save_dir = r"D:\Facial_Emotion\models\checkpoint_ResNet18"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(base_save_dir, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

print("Saving to:", run_dir)


# TRANSFORMS
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DATASET
data_path = r"D:\Facial_Emotion\data\processed_data"

full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
num_classes = len(full_dataset.classes)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# MODEL
model = models.resnet18(weights="DEFAULT")

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

# TRAIN SETTINGS
patience = 10
best_val_loss = float("inf")
early_stop_counter = 0
total_epochs = 40


# TRAIN LOOP
for epoch in range(total_epochs):

    # TRAIN 
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = 100 * correct / total

    #VALIDATION
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    progress = 100 * (epoch + 1) / total_epochs

    print(f"\nEpoch [{epoch+1}/{total_epochs}] ({progress:.1f}%)")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
    print("-" * 50)

    scheduler.step(val_loss)

    # ===== SAVE BEST =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0

        val_acc_percent = round(val_acc, 2)

        save_path = os.path.join(
            run_dir,
            f"best_epoch_{epoch+1}_valacc_{val_acc_percent}%.pth"
        )

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, save_path)

        print(f"model updated (Val Acc: {val_acc_percent}%)")

    else:
        early_stop_counter += 1
        print(f"Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered")
        break

print("Finished")
