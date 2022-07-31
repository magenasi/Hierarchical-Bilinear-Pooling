# coding=utf-8
"""
Author: xiezhenqing
date: 2022/7/31 11:17
desc: predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要将预测结果保存成txt，可以利用open打开txt文件，使用write方法写入txt，可以参考一下txt_annotation.py文件。
"""
import os
import time

from PIL import Image

from classification import Classification

classfication = Classification()

# # 检测单张图片
# while True:
# 	img = input('Input image filename:')
# 	try:
# 		image = Image.open(img)
# 	except:
# 		print('Open Error! Try again!')
# 		continue
# 	else:
# 		class_name = classfication.detect_image(image)
# 		print(class_name)

# 批量检测图片
img_dir = input('Input images dir path:')
try:
	images = os.listdir(img_dir)
	assert len(images) != 0
except:
	print('Open Error! Try again!')
else:
	for image in images:
		image = Image.open(os.path.join(img_dir, image))
		class_name, detect_time = classfication.detect_image(image)
		print('category: {}, detect_time: {:.5f} sec'.format(class_name, detect_time))
