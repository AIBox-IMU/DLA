import os
import random
import shutil

# 设置源目录和目标目录
source_images_dir = r'C:\data\dataset\images'  # 替换为images目录的路径
source_labels_dir = r'C:\data\dataset\labels'  # 替换为labels目录的路径

count = 100

target_base_dir = fr'C:\data\dataset-{count}'  # 替换为目标目录的路径

# 创建目标子目录
target_images_dir = os.path.join(target_base_dir, 'images')
target_labels_dir = os.path.join(target_base_dir, 'labels')

train_images_dir = os.path.join(target_images_dir, 'train')
val_images_dir = os.path.join(target_images_dir, 'val')
test_images_dir = os.path.join(target_images_dir, 'test')

train_labels_dir = os.path.join(target_labels_dir, 'train')
val_labels_dir = os.path.join(target_labels_dir, 'val')
test_labels_dir = os.path.join(target_labels_dir, 'test')

# 确保目标目录存在，不存在则创建
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# 获取源目录中的所有图片文件和标签文件
image_files = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg') or f.endswith('.png')]  # 假设图片为jpg或png格式
label_files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]  # 假设标签文件为txt格式

# 选择2000个随机的图片文件
selected_images = random.sample(image_files, count)

# 随机打乱顺序
random.shuffle(selected_images)

# 按照 8:1:1 的比例分配文件
train_images = selected_images[:int(count * 0.8)]  # 80% 用于训练集
val_images = selected_images[int(count * 0.8):int(count * 0.9)]  # 10% 用于验证集
test_images = selected_images[int(count * 0.9):]  # 10% 用于测试集

# 复制选中的图片和标签到目标目录
for image_file in train_images:
    label_file = image_file.rsplit('.', 1)[0] + '.txt'  # 假设标签文件与图片同名，后缀为.txt
    if label_file in label_files:
        shutil.copy(os.path.join(source_images_dir, image_file), os.path.join(train_images_dir, image_file))
        shutil.copy(os.path.join(source_labels_dir, label_file), os.path.join(train_labels_dir, label_file))

for image_file in val_images:
    label_file = image_file.rsplit('.', 1)[0] + '.txt'  # 假设标签文件与图片同名，后缀为.txt
    if label_file in label_files:
        shutil.copy(os.path.join(source_images_dir, image_file), os.path.join(val_images_dir, image_file))
        shutil.copy(os.path.join(source_labels_dir, label_file), os.path.join(val_labels_dir, label_file))

for image_file in test_images:
    label_file = image_file.rsplit('.', 1)[0] + '.txt'  # 假设标签文件与图片同名，后缀为.txt
    if label_file in label_files:
        shutil.copy(os.path.join(source_images_dir, image_file), os.path.join(test_images_dir, image_file))
        shutil.copy(os.path.join(source_labels_dir, label_file), os.path.join(test_labels_dir, label_file))

print("文件复制并分配完成！")
