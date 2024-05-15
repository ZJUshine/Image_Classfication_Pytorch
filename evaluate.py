# 用于评估模型的准确率和混淆矩阵
import torch
from torchvision import datasets, transforms, models
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_model

# 定义参数
MODEL_NAME = "" # 模型名称
DATASET_NAME = "" # 数据集名称
PRETRAIN = True # 是否使用预训练模型

# 模型路径
MODEL_PATH = f'./models/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_best.pth'
# 数据集路径
VAL_PATH = f'datasets/{DATASET_NAME}_process/val'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(),  
])

# 加载验证数据集
val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
CLASS_NUM = len(val_dataset.classes)
print("类别数：", CLASS_NUM)

# 定义模型
model = get_model(MODEL_NAME, PRETRAIN, CLASS_NUM)

# 检测是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备：",device)
# 加载模型
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# 预测并收集标签
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
conf_mat = confusion_matrix(all_labels, all_preds)
print("混淆矩阵:\n", conf_mat)

# 可视化混淆矩阵
plt.figure(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(f'./images/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_confusion_matrix.png')