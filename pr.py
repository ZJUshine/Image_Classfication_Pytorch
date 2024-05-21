import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import get_model
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

# 定义超参数
MODEL_NAME = "efficientnet" # 模型名称
DATASET_NAME = "flowers" # 数据集名称
PRETRAIN = False # 是否使用预训练模型
BATCH_SIZE = 10 # 批量大小

# 模型路径
MODEL_PATH = f'./models/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_best.pth'
# 数据集路径
DATASET_PATH = f'datasets/{DATASET_NAME}_process/val'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# 加载数据集
if DATASET_NAME == "CIFAR10":
    dataset = datasets.CIFAR10(root="./datasets", train=False, transform=transform, download=False)
elif DATASET_NAME == "StanfordCars":
    dataset = datasets.StanfordCars(root="./datasets", split="test", transform=transform, download=False)
elif DATASET_NAME == "Food101":
    dataset = datasets.Food101(root="./datasets", split="test", transform=transform, download=False)
elif DATASET_NAME == "CIFAR100":
    dataset = datasets.CIFAR100(root="./datasets", train=False, transform=transform, download=False)
else:
    # 加载自定义数据集
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

class_names = dataset.classes
CLASS_NUM = len(dataset.classes)

# 创建DataLoader来加载数据
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义模型
model = get_model(MODEL_NAME, PRETRAIN, CLASS_NUM)

# 检测是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)
# 加载模型
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# 收集所有预测和真实标签
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# 转换为numpy数组
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# 二值化标签
all_labels_bin = label_binarize(all_labels, classes=range(CLASS_NUM))

# 计算PR曲线
precision = dict()
recall = dict()
average_precision = dict()

for i in range(CLASS_NUM):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds[:, i])
    average_precision[i] = auc(recall[i], precision[i])

# 计算micro-average PR曲线
precision["micro"], recall["micro"], _ = precision_recall_curve(all_labels_bin.ravel(), all_preds.ravel())
average_precision["micro"] = auc(recall["micro"], precision["micro"])

# 绘制PR曲线
plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Micro-average Precision-Recall curve: AP={average_precision["micro"]:.2f}')
plt.savefig(f'./images/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_PR_curve.png')
plt.show()
