import torch
import random
import numpy as np
import json  # 用于保存类别映射
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm  # 增加进度条，便于观察训练进度

# --- 配置参数 ---
CONFIG = {
    "train_dir": "./data/pictures_train",
    "test_dir": "./data/pictures_test",
    "batch_size": 32,
    "img_size": 128,
    "epochs": 60,
    "lr": 0.001,
    "model_path": "MyGO_members_recognizer.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- 数据集定义 ---
class MyGODataSet(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"找不到路径: {self.root}")

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        for cls in self.classes:
            folder = self.root / cls
            for f in folder.iterdir():
                if f.suffix.lower() in valid_extensions:
                    self.samples.append((f, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"警告: 损坏的图片 {path}, 错误: {e}")
            # 返回一张全黑图避免训练中断
            img = Image.new('RGB', (CONFIG["img_size"], CONFIG["img_size"]))

        if self.transform:
            img = self.transform(img)
        return img, label


# --- 模型定义 ---
class MyGONet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),  # inplace节省显存
                nn.Conv2d(out_feat, out_feat, 3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            )

        self.features = nn.Sequential(
            block(3, 32),  # 128 -> 64
            block(32, 64),  # 64 -> 32
            block(64, 128),  # 32 -> 16
            block(128, 256)  # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --- 训练逻辑封装 ---
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += imgs.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in self.test_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += imgs.size(0)
        return total_loss / total, correct / total


# --- 主程序 ---
def main():
    set_seed(42)

    # 数据增强
    train_tf = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet标准
    ])

    test_tf = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据
    train_ds = MyGODataSet(CONFIG["train_dir"], train_tf)
    test_ds = MyGODataSet(CONFIG["test_dir"], test_tf)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # 保存类别映射，方便以后部署预测
    with open("class_indices.json", "w") as f:
        json.dump(train_ds.class_to_idx, f)

    model = MyGONet(len(train_ds.classes)).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)  # 更智能的调度

    writer = SummaryWriter("runs/mygo_experiment")
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, CONFIG["device"])

    best_acc = 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        t_loss, t_acc = trainer.train_epoch()
        v_loss, v_acc = trainer.test_epoch()

        # 日志
        writer.add_scalars("Loss", {"train": t_loss, "val": v_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": t_acc, "val": v_acc}, epoch)

        print(f"Epoch [{epoch}/{CONFIG['epochs']}] | Train Acc: {t_acc:.2%}, Val Acc: {v_acc:.2%}")

        # 调度器更新
        scheduler.step(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), CONFIG["model_path"])
            print(f"--> 最佳模型已保存 (Acc: {best_acc:.2%})")

    writer.close()
    print(f"\n训练完成! 最高准确率: {best_acc:.2%}")


if __name__ == "__main__":

    main()
