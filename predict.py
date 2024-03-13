# 用于预测模型的效果、展示预测结果、保存预测结果图片
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# 定义超参数
MODEL_NAME = "vit" # 模型名称
DATASET_NAME = "Howard_Cloud" # 数据集名称
PRETRAIN = True # 是否使用预训练模型
BATCH_SIZE = 10 # 取10张图片进行预测

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
    dataset = datasets.CIFAR10(root="./datasets",train=False, transform=transform,download=False)
elif DATASET_NAME == "StanfordCars":
    dataset = datasets.StanfordCars(root="./datasets",split="test", transform=transform,download=False)
elif DATASET_NAME == "Food101":
    dataset = datasets.Food101(root="./datasets",split="test", transform=transform,download=False)
elif DATASET_NAME == "CIFAR100":
    dataset = datasets.CIFAR100(root="./datasets",train=False, transform=transform,download=False)
else:
    # 加载自定义数据集
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

class_names = dataset.classes
CLASS_NUM = len(dataset.classes)

# 创建DataLoader来加载数据
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义模型
if MODEL_NAME == "Resnet50":
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, CLASS_NUM)
elif MODEL_NAME == "EfficientNet":
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, CLASS_NUM)
elif MODEL_NAME == "vgg16":
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, CLASS_NUM)
elif MODEL_NAME == "vgg19":
    model = models.vgg19(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, CLASS_NUM)
elif MODEL_NAME == "densenet169":
    model = models.densenet169(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, CLASS_NUM)
elif MODEL_NAME == "mobilenet_v3_small":
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, CLASS_NUM)
elif MODEL_NAME == "vit":
    model = models.vit_b_16(pretrained=False)
    model.heads.head = nn.Linear(model.heads.head.in_features, CLASS_NUM)


# 检测是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备：",device)
# 加载模型
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 随机选择10张图片
images, labels = zip(*random.sample(list(data_loader), 10))

# 预测并展示结果
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i, (img, label) in enumerate(zip(images, labels)):
    img = img.to(device)
    output = model(img)
    _, pred = torch.max(output, 1)
    pred = pred.cpu().numpy()[0]

    axes[i].imshow(np.transpose(img.cpu().numpy()[0], (1, 2, 0)))
    axes[i].set_title(f'Pred: {class_names[pred]}\nReal: {class_names[label[0]]}')
    axes[i].axis('off')

plt.savefig(f'./images/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_predict.png')
