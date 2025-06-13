import os
from pathlib import Path

import yaml


def calculate_area_ratios(label_file):
    """
    计算图像中每个目标的面积比例 (area_ratio)。

    参数:
        label_file (str): 标签文件路径（YOLO格式）
    返回:
        list: 每个目标的area_ratio列表
    """
    filtered_lines = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        areas = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            if cls in [3, 4, 12, 15, 16, 17]:
                continue
            w = float(parts[3])  # 目标宽度
            h = float(parts[4])  # 目标高度
            area = w * h
            areas.append(area)
            filtered_lines.append(line)
        sum_area = sum(areas)
        if sum_area == 0:
            return [], []  # 避免除以0
        area_ratios = [area / sum_area for area in areas]
        return area_ratios, filtered_lines


def filter_labels(label_file, condition):
    """
    根据条件筛选标签文件中的目标行。

    参数:
        label_file (str): 标签文件路径
        condition (callable): 用于筛选的条件函数，返回布尔值
    返回:
        list: 筛选后的标签行
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    area_ratios, filtered_lines = calculate_area_ratios(label_file)
    filtered_lines = [line for i, line in enumerate(filtered_lines) if condition(area_ratios[i])]

    return filtered_lines


def create_sub_dataset(original_images_dir, original_labels_dir, t1, t2, output_dir, condition, stage_name):
    """
    为特定目标尺寸阶段创建子数据集。

    参数:
        original_images_dir (str): 原始图像目录路径
        original_labels_dir (str): 原始标签目录路径
        t1 (float): 小目标与中目标的阈值
        t2 (float): 中目标与大目标的阈值
        output_dir (str): 子数据集输出目录
        condition (callable): 筛选条件函数
        stage_name (str): 阶段名称（big、medium、small）
    """
    # 创建子数据集目录结构
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images', stage_name)
    labels_dir = os.path.join(output_dir, 'labels', stage_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 遍历原始标签文件
    for label_file in os.listdir(original_labels_dir):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')  # 假设图像为.jpg格式
            original_image_path = os.path.join(original_images_dir, image_file)
            original_label_path = os.path.join(original_labels_dir, label_file)

            # 筛选标签行
            filtered_lines = filter_labels(original_label_path, condition)
            if not filtered_lines:
                continue

            # 保存筛选后的标签文件
            new_label_path = os.path.join(labels_dir, label_file)

            with open(new_label_path, 'w') as f:
                f.writelines(filtered_lines)

            # 使用符号链接关联原始图像
            new_image_path = os.path.join(images_dir, image_file)
            try:
                os.symlink(original_image_path, new_image_path)
            except FileExistsError:
                continue
    # 创建data.yaml文件（用于YOLO训练）
    data_yaml = {
        'train': f'C:\\data\\dataset-cl-k4\\{Path(output_dir).stem}\\images\\train',
        'val': f'C:\\data\\dataset-cl-k4\\{Path(output_dir).stem}\\images\\val',  # 假设验证集与训练集相同，可根据需要调整
        'nc': 16,  # 类别数，需根据实际情况设置
        'names': [
            "t", "t1", "t2", "t3", "t4", "paragraph",
            "author", "keyword", "abstract", "reference",
            "graph", "note", "other", "formula", "table",
            "footnote"
        ]
    }
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)


# 主程序
if __name__ == "__main__":
    # 设置原始数据集路径（请替换为实际路径）
    original_images_dir = Path(r'C:\data\dataset-8000\images')
    original_labels_dir = Path(r'C:\data\dataset-8000\labels')
    target_dir = Path(r'C:\data\dataset-cl-k4')
    # 设置阈值（请替换为您计算出的t1和t2）
    # t1 = 0.1278  # 小目标与中目标的阈值（示例值）
    # t2 = 0.4286  # 中目标与大目标的阈值（示例值）
    t1 = 0.1894  # 小目标与中目标的阈值（示例值）
    t2 = 0.2429  # 中目标与大目标的阈值（示例值）
    t3=0.4924
    # 定义筛选条件
    conditions = {
        '1': lambda ratio: ratio >= t3,  # 大目标：area_ratio > t2
        '2': lambda ratio: ratio >= t2,  # 中目标：t1 < area_ratio ≤ t2
        '3': lambda ratio: ratio >= t1,  # 小目标：area_ratio ≤ t1
        '4': lambda ratio: ratio > 0,  # 小目标：area_ratio ≤ t1
    }

    # 生成三个子数据集
    for stage, condition in conditions.items():
        for stage_name in ['train', 'val', 'test']:
            print(f"正在生成 {stage} 目标数据集...")
            create_sub_dataset(original_images_dir / stage_name, original_labels_dir / stage_name,
                               t1, t2, target_dir / f'{stage}_target_dataset',
                               condition,
                               stage_name)
        print("子数据集生成完成！")
