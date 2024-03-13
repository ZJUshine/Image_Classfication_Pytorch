# Image_Classfication_Pytorch

# Train

```
python train.py --model_name Resnet50 --dataset_name CIFAR100 --pretrain True --epochs 20
```

支持多种模型和自带数据集以及自定义数据集

![train](show_images/flowers_Resnet50_Pretrained_False_result.png)

# Predict

```
python predict.py
```

取若干张进行预测

![predict](show_images/flowers_Resnet50_Pretrained_False_predict.png)

# Evaluate

```
python evaluate.py
```

构造混淆矩阵

![evaluate](show_images/flowers_Resnet50_Pretrained_False_confusion_matrix.png)

# Qt

```
python qt.py
```

使用Pyqt5来构造可视化界面

![qt](show_images/qt.png)

# Web

```
python web.py
```

使用Gradio来构造Web界面

![web](show_images/web.png)