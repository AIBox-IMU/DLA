import json
import shutil
from pathlib import Path


def filter_aca(json_path, data_type):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for img in data['images']:
            if not img['file_name'].startswith('scientific'):
                continue
            img_id = img['id']
            write_path = r'C:\data\D4LA-acadamic\{}\labels\{}'.format(data_type,
                                                                      img['file_name'].replace('.png', '.txt'))
            with open(write_path, 'w', encoding='utf-8') as f2:
                for annotation in data['annotations']:
                    if annotation['image_id'] == img_id:
                        box = annotation['bbox']
                        category_id = annotation['category_id']
                        x_center = (box[0] + box[2] / 2) / img['width']
                        y_center = (box[1] + box[3] / 2) / img['height']
                        width = box[2] / img['width']
                        height = box[3] / img['height']
                        line = ' '.join([str(category_id - 1), str(x_center), str(y_center), str(width), str(height)])
                        f2.write(line + '\n')

                        ori_pic_path = fr"C:\data\D4LA\D4LA\D4LA\{data_type}_images\{img['file_name']}"
                        tar_pic_path = fr"C:\data\D4LA-acadamic\{data_type}\images\{img['file_name']}"
                        shutil.copy(ori_pic_path, tar_pic_path)


if __name__ == '__main__':
    json_path = Path(r'C:\data\D4LA\D4LA\D4LA\json\train.json')
    filter_aca(json_path, json_path.stem)
