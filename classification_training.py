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
EPOCHS = 100
LR = 0.0001
IMG_SIZE = 224
WEIGHT_DECAY = 0.0001  # L2 Regularization 개념


# 노이즈 추가 함수(증강 때 사용)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# Label Smoothing 적용한 BCE Loss
class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.loss_fn(inputs, targets)


# 학습 데이터셋 증강 파트
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    # 밤 환경 고려 증강 기법
    transforms.ColorJitter(  
        brightness = (0.3, 1.0),
        contrast = (0.5, 1.5),
        saturation = (0.3, 1.2)
    ),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.02),  # 노이즈 추가 
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
model.fc = nn.Sequential(nn.Dropout(p = 0.8),  # Dropout 적용
                         model.fc)  # Sigmoid 제거

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = SmoothBCEWithLogitsLoss(smoothing = 0.15)  # 0.1 적용 시 과적합 그래도 심함. 0.15로 적용.
optimizer = optim.AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)  # Weight Decay 적용.


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
        torch.save(model.state_dict(), 'xception_model_relieve_overfitting.pth')

# 기록 저장 및 시각화
results = pd.DataFrame({
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
}).add_prefix('xception_')

results.to_csv("xception_model_relieve_overfitting.csv", index=False)

best_epoch = results['xception_val_accuracy'].idxmax() + 1
best_acc = results['xception_val_accuracy'].max()

print('best epoch: ', best_epoch)
print('best accuracy: ', best_acc)

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