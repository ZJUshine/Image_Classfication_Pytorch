# 用于处理数据集，将数据集划分为训练集和验证集
import os
import shutil
import random
import glob

# 数据集文件夹名称
DATASET_NAME = ""

# 数据集文件夹路径
data_dir = f"./datasets/{DATASET_NAME}"
# 训练集和验证集文件夹路径
train_dir = f"./datasets/{DATASET_NAME}_process/train"
val_dir = f"./datasets/{DATASET_NAME}_process/val"
# 训练集和验证集的比例
split_ratio = 0.8

# 创建训练集和验证集文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取子文件夹列表
class_folders = glob.glob(os.path.join(data_dir, "*"))

for class_folder in class_folders:
    # 获取当前类别的图像文件列表
    image_files = glob.glob(os.path.join(class_folder, "*"))
    
    # 随机打乱图像文件列表
    random.shuffle(image_files)
    
    # 计算训练集和验证集的分割点
    split_point = int(len(image_files) * split_ratio)
    
    # 将图像文件复制到相应的目标文件夹
    for i, image_file in enumerate(image_files):
        if i < split_point:
            target_folder = os.path.join(train_dir, os.path.basename(class_folder))
        else:
            target_folder = os.path.join(val_dir, os.path.basename(class_folder))
        
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy(image_file, target_folder)
        
    print(f"Processed {class_folder}: {split_point} images for training, {len(image_files) - split_point} images for validation")
