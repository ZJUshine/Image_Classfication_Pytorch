import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def count_images_in_subfolders(folder_path):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    folder_image_count = {}

    # 遍历主文件夹中的所有子文件夹
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            image_count = 0
            # 遍历子文件夹中的所有文件
            for file_name in os.listdir(subfolder_path):
                file_extension = os.path.splitext(file_name)[1].lower()
                if file_extension in image_extensions:
                    image_count += 1
            folder_image_count[subfolder_name] = image_count

    return folder_image_count

def plot_image_counts(image_counts):
    # 设置中文字体
    font_path = './SimHei.ttf'
    font_prop = FontProperties(fname=font_path)

    subfolders = list(image_counts.keys())
    counts = list(image_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(subfolders, counts, color='skyblue')
    plt.xlabel('垃圾类别', fontproperties=font_prop)
    plt.ylabel('图片数量', fontproperties=font_prop)
    plt.title('每个垃圾类别中的图片数量', fontproperties=font_prop)
    plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("datasets_overview.png")

# 设置主文件夹路径
main_folder_path = './datasets/trash'

# 统计每个子文件夹中的图片数量
image_counts = count_images_in_subfolders(main_folder_path)

# 绘制柱状图
plot_image_counts(image_counts)
