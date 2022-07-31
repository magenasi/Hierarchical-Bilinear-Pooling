"""
根据数据集将类别名称写入 cls_classes.txt 文件中
"""

import os


cls_classes = os.listdir(os.path.join('..', 'datasets', 'train'))
for cls_class in cls_classes:
    with open('cls_classes.txt', 'a+') as f:
        f.write(cls_class + '\n')

print('Done!')
