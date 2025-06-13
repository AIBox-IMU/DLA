import os
import traceback
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO


def cuda_available():
    import torch
    return torch.cuda.is_available()


def train(model_path, train_yaml, dataset, pre_weight):
    model = YOLO(model_path)
    model.load(pre_weight)
    model.train(
        # data=rf'C:\PycharmProjects\ultralytics-yolo11-0214\dataset\{dataset}.yaml',
        data=train_yaml,
        cache=False,
        imgsz=640,
        epochs=300,
        batch=32,
        close_mosaic=0,
        workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
        device='0',  # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
        optimizer='SGD',  # using SGD
        patience=50,  # set 0 to close earlystop.
        # resume=True, # 断点续训,YOLO初始化时选择last.pt
        amp=False, # close amp | loss出现nan可以关闭amp
        # fraction=0.2,
        project='runs/train-light',
        name=f'{model_path.stem}-{dataset}',
    )


def mul_train(dataset):
    yaml_list = [
        # 'yolo11m.yaml',
        # 'yolo11m-CGRFPN.yaml',
        # 'yolo11m-KernelWarehouse.yaml',
        # 'yolo11m-doc.yaml',
        # 'yolo11-mobilenetv4.yaml',
        # "yolo11-CGAFusion.yaml",
        # "yolo11-mobilenetv4-CGAFusion.yaml",
        # "yolo11-efficientViT.yaml",
        # "yolo11-fasternet.yaml",
        # "yolo11-repvit.yaml"
        'yolo11m-CGAFusion.yaml'
    ]

    train_yaml = r'C:\data\dataset-cl\small_target_dataset\data.yaml'
    yaml_path = Path(r'C:\PycharmProjects\ultralytics-yolo11-0214\ultralytics\cfg\models\4-light')
    for file in yaml_list:
        try:
            model_path = Path(r'C:\PycharmProjects\ultralytics-yolo11-0214\ultralytics\cfg\models\11')
            train(model_path / file, train_yaml, dataset, r'./yolo11n.pt')
        except Exception as e:
            print(f'file_name:{file}, error:{e}')
            print(traceback.format_exc())


def train_by_dir(dataset):
    yaml_path = Path(r'C:\PycharmProjects\ultralytics-yolo11-0214\ultralytics\cfg\models\3-improve')
    # yaml_path = Path(r'ultralytics/cfg/models/11')
    mapper = [
        (r'./yolo11n.pt', "yolo11-KW-SL.yaml"),
        # ('./yolo11m.pt', 'yolo11m-slimneck.yaml')
        # (r'./yolo11n.pt', "yolo11-doc.yaml"),
        # (r'./yolo11n.pt', "yolo11-mobilenetv4-CGAFusion.yaml"),
        # (r'./yolo11m.pt', "yolo11m.yaml"),
        # (r'./yolo11m.pt', "yolo11m-doc.yaml"),
        # (r'./yolo11m.pt', "yolo11m-mobilenetv4-CGAFusion.yaml"),
        # (r'./yolo11s.pt', "yolo11s-doc.yaml"),
        # (r'./yolo11s.pt', "yolo11s.yaml"),
        # (r'./yolo11s.pt', "yolo11s-mobilenetv4-CGAFusion.yaml"),
        # (r'./yolo11n.pt', 'yolo11-dyhead-DCNV4.yaml')
    ]
    for weight_path, yaml_file in mapper:
        train(yaml_path / yaml_file, r'C:\data\dataset-cl\small_target_dataset\data.yaml', dataset, weight_path)


def train_cl():
    # yaml_path = Path(r'C:\PycharmProjects\ultralytics-yolo11-0214\ultralytics\cfg\models\4-light')
    # mul_train('d4la')
    # train_by_dir()
    model_path = r'C:\PycharmProjects\ultralytics-yolo11-0214\ultralytics\cfg\models\11\yolo11.yaml'
    # weight_path = r'C:\PycharmProjects\ultralytics-yolo11-0214\runs\train\yolo11-medium_target_dataset\weights\best.pt'
    _weight_path = r'C:\PycharmProjects\ultralytics-yolo11-0214\runs\train\yolo11-{}\weights\best.pt'

    mapper = [
        # ('./yolo11n.pt', r'C:\data\dataset-cl-k4\1_target_dataset\data.yaml', '1_target_dataset'),
        # (_weight_path.format('1_target_dataset'), r'C:\data\dataset-cl-k4\2_target_dataset\data.yaml',
        #  '2_target_dataset'),
        # (_weight_path.format('2_target_dataset'), r'C:\data\dataset-cl-k4\3_target_dataset\data.yaml',
        #  '3_target_dataset'),
        # (_weight_path.format('3_target_dataset'), r'C:\data\dataset-cl-k4\4_target_dataset\data.yaml',
        #  '4_target_dataset'),
        # (r'./yolo11n.pt', r'C:\data\dataset-cl\small_target_dataset\data.yaml', 'small_target_dataset'),
        (r'./yolo11n.pt', "yolo11-CGAFusion.yaml", 'small_target_dataset'),
        (r'./yolo11n.pt', "yolo11-efficientViT.yaml", 'small_target_dataset'),
        (r'./yolo11n.pt', "yolo11-fasternet.yaml", 'small_target_dataset'),
        (r'./yolo11n.pt', "yolo11-mobilenetv4.yaml", 'small_target_dataset'),
        (r'./yolo11n.pt', "yolo11-mobilenetv4-CGAFusion.yaml", 'small_target_dataset'),
        (r'./yolo11n.pt', "yolo11-repvit.yaml", 'small_target_dataset'),

    ]
    train_yaml = r'C:\data\dataset-cl\small_target_dataset\data.yaml'
    for weight_path, yaml_path, dataset in mapper:
        train(Path(yaml_path), train_yaml, dataset, weight_path)


if __name__ == '__main__':
    train_by_dir('small_target_dataset')
