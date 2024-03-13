# 实现了一个简单的QT图形界面，可以选择模型和图片，然后进行识别
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        uic.loadUi('main.ui', self) # 加载ui文件

        self.DATASET_NAME = "flowers" # 数据集名称
        self.MODEL_NAME = "Resnet50" # 模型名称
        datasets_path = f'datasets/{self.DATASET_NAME}_process/train' # 数据集路径

        # 获取路径下所有的labels
        self.class_names = [d.name for d in os.scandir(datasets_path) if d.is_dir()]
        print(self.class_names)
        self.CLASS_NUM = len(self.class_names)

        self.model = None
        self.image = None
        
        # 检测是否有可用的GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")

        self.pushButton_model.clicked.connect(self.select_model) # 建立点击事件与选择模型函数连接
        self.pushButton_image.clicked.connect(self.select_image) # 建立点击事件与选择图片函数连接
        self.pushButton_start.clicked.connect(self.start_recognize) # 建立点击事件与开始识别函数连接

    def select_model(self):
        self.modelpath, _ = QFileDialog.getOpenFileName(self, "选择模型") # 打开文件对话框来选择模型 
        if self.MODEL_NAME == "Resnet50":
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.CLASS_NUM)
        elif self.MODEL_NAME == "EfficientNet":
            self.model = models.efficientnet_b3(pretrained=False)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.CLASS_NUM)
        elif self.MODEL_NAME == "vgg16":
            self.model = models.vgg16(pretrained=False)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, self.CLASS_NUM)
        elif self.MODEL_NAME == "vgg19":
            self.model = models.vgg19(pretrained=False)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, self.CLASS_NUM)
        elif self.MODEL_NAME == "densenet169":
            self.model = models.densenet169(pretrained=False)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.CLASS_NUM)
        elif self.MODEL_NAME == "mobilenet_v3_small":
            self.model = models.mobilenet_v3_small(pretrained=False)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.CLASS_NUM)
        elif self.MODEL_NAME == "vit":
            self.model = models.vit_b_16(pretrained=False)
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.CLASS_NUM)

        # 加载模型
        self.model.load_state_dict(torch.load(self.modelpath, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def select_image(self):
        self.imagepath, _ = QFileDialog.getOpenFileName(self, "选择图片") # 打开文件对话框来选择图片
        if self.imagepath:
            self.image_display = QPixmap(self.imagepath)
            # 将图片大小调整为Label控件的大小
            self.image_display = self.image_display.scaled(self.label_image.width(), self.label_image.height())
            self.label_image.setPixmap(self.image_display) # 显示图片在对应的Label控件中
            self.image = Image.open(self.imagepath) # 读取图片

    def start_recognize(self):
        if self.model and self.image:
            # 数据预处理
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
            image = transform(self.image) # 读取图片
            image = image.unsqueeze(0)
            image = image.to(self.device)
            output = self.model(image)
            _, predicted = torch.max(output.data, 1)
            result = self.class_names[predicted[0].item()]
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted[0]].item()
            self.label_result.setText(result) # 将结果显示在对应的Label控件中
            self.label_conf.setText(str(round(confidence, 3))) # 将置信度显示在对应的Label控件中

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()