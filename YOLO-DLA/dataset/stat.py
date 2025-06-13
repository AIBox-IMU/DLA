import os
from collections import defaultdict
from glob import glob


def calculate_page_area_distribution(labels_dir):
    """
    统计每个类别目标的总面积占比（总目标面积/总页面面积）和平均目标占比（平均每个目标的面积占比）
    :param labels_dir: YOLO标签文件目录
    :return: 统计字典 {class_id: {'total_area_%': 总占比, 'avg_area_%': 平均占比, 'count': 目标数}}, 总页面数
    """
    # 初始化统计字典
    class_area = defaultdict(float)  # 按类别累计面积占比
    class_count = defaultdict(int)  # 按类别累计目标数量
    total_pages = 0  # 总页面数（以标签文件数量为准）

    # 获取所有标签文件
    label_files = glob(os.path.join(labels_dir, "*.txt"))
    total_pages = len(label_files)

    # 遍历标签文件
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析YOLO格式
                try:
                    parts = list(map(float, line.split()))
                    if len(parts) < 5:
                        raise ValueError("Invalid line format")

                    class_id = int(parts[0])
                    if class_id in [3, 4, 12, 15, 17]:
                        continue

                    w = parts[3]
                    h = parts[4]

                    # 计算当前目标的面积占比（归一化的w*h即代表占页面面积的比例）
                    area_percent = w * h * 100  # 转为百分比

                    # 更新统计
                    class_area[class_id] += area_percent
                    class_count[class_id] += 1

                except (ValueError, IndexError) as e:
                    print(f"跳过无效行: {line} (文件: {os.path.basename(label_file)}), 错误: {str(e)}")

    # 计算总占比和平均占比
    stats = {}
    for class_id in class_area:
        total_area_percent = class_area[class_id] / total_pages  # 总占比 = 累计面积 / 总页面数
        avg_area_percent = class_area[class_id] / class_count[class_id]  # 平均占比 = 累计面积 / 目标数
        stats[class_id] = {
            'total_area_%': round(total_area_percent, 2),
            'avg_area_%': round(avg_area_percent, 2),
            'count': class_count[class_id]
        }

    print(stats)
    return stats, total_pages


def print_distribution(stats, total_pages):
    """ 打印统计结果 """
    print(f"\n{'Class':<6} {'Total Area (%)':<15} {'Avg Area (%)':<15} {'Count':<10}")
    print("-" * 50)
    for class_id in sorted(stats.keys()):
        info = stats[class_id]
        print(f"{class_id:<6} {info['total_area_%']:<15} {info['avg_area_%']:<15} {info['count']:<10}")
    print("-" * 50)
    print(f"Total Pages: {total_pages}")


import matplotlib.pyplot as plt
import numpy as np

# 类别名称映射
class_names = {
    0: "文章标题",
    1: "一级标题",
    2: "二级标题",
    5: "正文段落",
    6: "作者",
    7: "关键词",
    8: "摘要",
    9: "参考文献",
    10: "图片",
    11: "图注表注",
    13: "公式",
    14: "表格"
}

class_names = {
    "文章标题": 'blue',
    "一级标题": 'blue',
    "二级标题": 'blue',
    "正文段落": 0,
    "作者": 0,
    "关键词": 0,
    "摘要": 0,
    "参考文献": 0,
    "图片": 0,
    "图注表注": 0,
    "公式": 0,
    "表格": 0,
}


def plot_class_distribution(stats):
    # 准备数据
    sorted_classes = sorted(stats.keys(), key=lambda x: stats[x]['total_area_%'], reverse=True)
    labels = [class_names[c] for c in sorted_classes]
    values = [stats[c]['total_area_%'] for c in sorted_classes]

    # 创建画布
    plt.figure(figsize=(12, 8))

    # 绘制水平条形图（对数刻度）
    bars = plt.barh(labels, values, color='#2c7bb6', edgecolor='black', log=True)

    # 自动调整标签显示
    plt.gca().invert_yaxis()  # 数值大的在上方
    plt.xlabel('各类版面元素面积占比(%) - 对数刻度', fontsize=12)
    plt.title('文档各类版面元素面积分布统计', fontsize=14, pad=20)

    # 添加数值标签（自动跳过过小的值）
    for bar in bars:
        width = bar.get_width()

        if width == 0.08:
            plt.text(width * 1.5, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%',
                     va='center', ha='left', fontsize=10)
        elif width <= 0.04:
            plt.text(width * 2.7, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%',
                     va='center', ha='left', fontsize=10)
        else:
            plt.text(width * 1.05, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%',
                     va='center', ha='left', fontsize=10)

    # 突出显示特殊类别
    highlight_classes = ["table", "formula", "graph"]
    for idx, label in enumerate(labels):
        if label in highlight_classes:
            bars[idx].set_color('#d7191c')  # 红色突出显示
            plt.text(0.5, bars[idx].get_y() + bars[idx].get_height() / 2,
                     "← Key Elements",
                     color='#d7191c', va='center', ha='left', fontsize=10)

    # 优化坐标轴
    plt.xlim(0.1, max(values) * 1.5)  # 留出标注空间
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    stats = {5: {'total_area_%': 40.77, 'avg_area_%': 6.33, 'count': 115855},
             1: {'total_area_%': 0.35, 'avg_area_%': 0.72, 'count': 8687},
             2: {'total_area_%': 0.67, 'avg_area_%': 0.68, 'count': 17772},
             10: {'total_area_%': 8.76, 'avg_area_%': 15.27, 'count': 10322},
             11: {'total_area_%': 1.32, 'avg_area_%': 1.66, 'count': 14308},
             13: {'total_area_%': 3.89, 'avg_area_%': 3.0, 'count': 23347},
             9: {'total_area_%': 1.71, 'avg_area_%': 30.8, 'count': 1001},
             6: {'total_area_%': 0.25, 'avg_area_%': 6.5, 'count': 700},
             0: {'total_area_%': 0.08, 'avg_area_%': 4.07, 'count': 353},
             14: {'total_area_%': 2.98, 'avg_area_%': 12.34, 'count': 4349},
             7: {'total_area_%': 0.04, 'avg_area_%': 2.44, 'count': 331},
             8: {'total_area_%': 0.19, 'avg_area_%': 14.34, 'count': 239}}
    # 假设stats是之前计算的统计字典

    from pylab import mpl

    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    plot_class_distribution(stats)
