# train.py:
# 1.训练模型
# 2.保存模型最优和最后的权重
# 3.绘制损失、准确率数据和结果图像
# 4.支持tensorboard可视化
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils import get_model

# 用argparse来解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="resnet", help="模型名称")
parser.add_argument("--dataset_name", type=str, default="trash", help="数据集名称")
parser.add_argument("--pretrain", type=bool, default=False, help="是否加载预训练模型")
parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
args = parser.parse_args()

# 定义超参数
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = args.epochs
MODEL_NAME = args.model_name
DATASET_NAME = args.dataset_name
PRETRAIN = args.pretrain

# 创建models文件夹来保存模型权重
os.makedirs(f"models/{DATASET_NAME}_{MODEL_NAME}",exist_ok = True)

# 创建images文件夹来保存结果
os.makedirs(f"images/{DATASET_NAME}_{MODEL_NAME}",exist_ok = True)

# 创建一个SummaryWriter实例
writer = SummaryWriter(f'runs/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_experiment')

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),  # 随机裁剪并缩放到224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.ToTensor(),  # 将图像转换为Tensor
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),  # 缩放到224x224
    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备：",device)

# 加载默认数据集
if DATASET_NAME == "CIFAR10":
    train_dataset = datasets.CIFAR10(root="./datasets",train=True, transform=val_transform,download=True)
    val_dataset = datasets.CIFAR10(root="./datasets",train=False, transform=val_transform,download=True)
elif DATASET_NAME == "StanfordCars":
    train_dataset = datasets.StanfordCars(root="./datasets",split="train", transform=val_transform,download=False)
    val_dataset = datasets.StanfordCars(root="./datasets",split="test", transform=val_transform,download=False)
elif DATASET_NAME == "Food101":
    train_dataset = datasets.Food101(root="./datasets",split="train", transform=val_transform,download=True)
    val_dataset = datasets.Food101(root="./datasets",split="test", transform=val_transform,download=True)
elif DATASET_NAME == "CIFAR100":
    train_dataset = datasets.CIFAR100(root="./datasets",train=True, transform=val_transform,download=True)
    val_dataset = datasets.CIFAR100(root="./datasets",train=False, transform=val_transform,download=True)
else:
    # 加载自定义数据集
    train_path = f'datasets/{DATASET_NAME}_process/train'
    val_path = f'datasets/{DATASET_NAME}_process/val'
    train_dataset = datasets.ImageFolder(train_path,transform = train_transform)
    val_dataset = datasets.ImageFolder(val_path,transform = val_transform)

print("训练集大小：",len(train_dataset))
print("验证集大小：",len(val_dataset))
print("类别：",train_dataset.classes)
print("类别对应的索引：",train_dataset.class_to_idx)
CLASS_NUM = len(train_dataset.classes)
print("类别数量为",CLASS_NUM)
# 加载train和val数据集
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Model_name:",MODEL_NAME)

# 加载不同种类的模型
transfer_model = get_model(MODEL_NAME, PRETRAIN, CLASS_NUM)
transfer_model = transfer_model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transfer_model.parameters(), lr=LEARNING_RATE)

# 初始化最佳准确率、训练损失、验证损失、训练准确率和验证准确率
best_accuracy = 0.0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
# 训练模型
for epoch in range(EPOCHS):
    transfer_model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = transfer_model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100 * train_correct / len(train_loader.dataset)

    # 验证模型
    transfer_model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = transfer_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * val_correct / len(val_loader.dataset)


    # 仅当准确率提高时保存模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(transfer_model.state_dict(), f'./models/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_best.pth')

    # 保存损失和准确率数据
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    # 记录训练和验证损失及准确率
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# 保存模型
torch.save(transfer_model.state_dict(), f'./models/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_epoch_{EPOCHS}.pth')

# 关闭writer
writer.close()

# 绘制损失图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'{DATASET_NAME}_{MODEL_NAME} Training and Validation Loss')
plt.legend()

# 绘制准确率图
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title(f'{DATASET_NAME}_{MODEL_NAME} Training and Validation Accuracy')
plt.legend()
plt.savefig(f"./images/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_result.png")