# 使用gradio库搭建web应用
import torch
from torchvision import transforms, models
from torch import nn
import gradio as gr
import os
from utils import get_model

MODEL_NAME = "" # 模型名称
DATASET_NAME = "" # 数据集名称
PRETRAIN = True # 是否使用预训练模型
datasets_path = f'datasets/{DATASET_NAME}_process/train' # 数据集路径
# 获取路径下所有的labels
labels = [d.name for d in os.scandir(datasets_path) if d.is_dir()]
CLASS_NUM = len(labels)
# 检测是否有可用的GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 导入模型
MODEL_PATH = f'./models/{DATASET_NAME}_{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_Pretrained_{PRETRAIN}_best.pth'

model = get_model(MODEL_NAME, PRETRAIN, CLASS_NUM)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def predict(input):
  # 预处理图像归一化
  input = transform(input).unsqueeze(0).to(device)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(input)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(CLASS_NUM)}
  return confidences

predict_app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="原始图像"),
    outputs=gr.Label(num_top_classes=5),
    title="图像分类",
    description="上传一张图片，模型将预测图片的类别。"
)
predict_app.launch(share = True)