import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import timm

# 미세조정 파라미터
BATCH_SIZE = 32
EPOCHS = 300
LR = 0.0001
IMG_SIZE = 224


# 학습 데이터셋 증강 파트
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = ImageFolder('classification_data/train', transform=transform_train)
val_dataset = ImageFolder('classification_data/test', transform=transform_val)

# 클래스 확인
print("클래스 인덱스 확인:", train_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 사전 학습 모델
model = timm.create_model('xception', pretrained=True, num_classes=1)
model.fc = nn.Sequential(model.fc, nn.Sigmoid())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

print(f"Using device: {device}")

train_losses = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0


for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", leave = False, dynamic_ncols=True):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    train_loss = running_train_loss / len(train_loader)
    val_loss = running_val_loss / len(val_loader)
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 최고 성능 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_xception_model.pth')

# 기록 저장 및 시각화
results = pd.DataFrame({
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
}).add_prefix('xception_')

results.to_csv("history_xception.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(results.index + 1, results['xception_train_loss'], label='Train Loss')
plt.plot(results.index + 1, results['xception_val_loss'], label='Val Loss')
plt.plot(results.index + 1, results['xception_val_accuracy'], label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.title("xception - Train & Validation")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('xception_output.png')

plt.show()