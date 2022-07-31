# coding=utf-8
"""
Author: xiezhenqing
date: 2022/7/31 2:05
desc:
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from torch import nn

from nets.hbp_model import HBPNet
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input, show_config)


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
# --------------------------------------------#
class Classification(object):
	_defaults = {
		# --------------------------------------------------------------------------#
		#   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
		#   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
		#   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
		# --------------------------------------------------------------------------#
		"model_path": 'logs/best_epoch_weights.pth',
		"classes_path": 'model_data/cls_classes.txt',
		# --------------------------------------------------------------------#
		#   输入的图片大小
		# --------------------------------------------------------------------#
		"input_shape": [224, 224],
		# --------------------------------------------------------------------#
		#   所用模型种类、注意力模块、投射层输出通道数：
		# --------------------------------------------------------------------#
		"backbone": 'MobileNetV3Small',
		"phi": 2,
		"proj_k": 8192,
		# --------------------------------------------------------------------#
		#   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
		#   否则对图像进行CenterCrop
		# --------------------------------------------------------------------#
		"letterbox_image": False,
		# -------------------------------#
		#   是否使用Cuda
		#   没有GPU可以设置成False
		# -------------------------------#
		"cuda": True
	}

	@classmethod
	def get_defaults(cls, n):
		if n in cls._defaults:
			return cls._defaults[n]
		else:
			return "Unrecognized attribute name '" + n + "'"

	# ---------------------------------------------------#
	#   初始化classification
	# ---------------------------------------------------#
	def __init__(self, **kwargs):
		self.__dict__.update(self._defaults)
		for name, value in kwargs.items():
			setattr(self, name, value)

		# ---------------------------------------------------#
		#   获得种类
		# ---------------------------------------------------#
		self.class_names, self.num_classes = get_classes(self.classes_path)
		self.generate()

		show_config(**self._defaults)

	# ---------------------------------------------------#
	#   获得所有的分类
	# ---------------------------------------------------#
	def generate(self):
		# ---------------------------------------------------#
		#   载入模型与权值
		# ---------------------------------------------------#
		self.model = HBPNet(backbone=self.backbone, phi=self.phi, pretrained=False, num_classes=self.num_classes, proj_k=self.proj_k)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.load_state_dict(torch.load(self.model_path, map_location=device))
		self.model = self.model.eval()
		print('{} model, and classes loaded.'.format(self.model_path))

		if self.cuda:
			self.model = self.model.cuda()

	# ---------------------------------------------------#
	#   检测图片
	# ---------------------------------------------------#
	def detect_image(self, image):
		start_time = time.time()
		# ---------------------------------------------------------#
		#   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
		#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
		# ---------------------------------------------------------#
		image = cvtColor(image)
		# ---------------------------------------------------#
		#   对图片进行不失真的resize
		# ---------------------------------------------------#
		image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]],
		                             self.letterbox_image)
		# ---------------------------------------------------------#
		#   归一化+添加上batch_size维度+转置
		# ---------------------------------------------------------#
		image_data = np.transpose(
			np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

		with torch.no_grad():
			photo = torch.from_numpy(image_data)
			if self.cuda:
				photo = photo.cuda()
			# ---------------------------------------------------#
			#   图片传入网络进行预测
			# ---------------------------------------------------#
			preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
		# ---------------------------------------------------#
		#   获得所属种类
		# ---------------------------------------------------#
		class_name = self.class_names[np.argmax(preds)]
		probability = np.max(preds)

		end_time = time.time()
		detect_time = end_time - start_time

		# ---------------------------------------------------#
		#   绘图并写字
		# ---------------------------------------------------#
		plt.subplot(1, 1, 1)
		plt.imshow(np.array(image))
		plt.title('Class:%s Probability:%.3f' % (class_name, probability))
		plt.show()
		return class_name, detect_time
