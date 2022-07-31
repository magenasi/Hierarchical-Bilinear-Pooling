# coding=utf-8

"""
Author: xiezhenqing
Email: 441798804@qq.com
date: 2022/7/28 11:05
desc: 全连接层参数太多，训练和推理都由困难
"""

import torch
import torch.nn as nn

from resnet import resnet50
from nets.attention import se_block, cbam_block, eca_block, CA_Block

attention_block = [se_block, cbam_block, eca_block, CA_Block]


class ResNetWithAttenBili(nn.Module):
	"""
	在 ResNet50 基础上从 stage2: (N, 512, 28, 28)
					   stage3: (N, 1024, 14, 14)
					   stage4: (N, 2048, 7, 7)
	三分支分别接一个注意力机制模块，后经过经典双线性汇合(bilinear-cnn)生成三个输出向量
	因参数过多，冻结骨干网络参数训练
	"""
	def __init__(self,
				 phi=0,
				 numclasses=1000,
				 pretrained=False):
		super(ResNetWithAttenBili, self).__init__()
		self.phi = phi
		self.backbone = resnet50(pretrained)

		self.fc1 = torch.nn.Linear(512**2, numclasses)
		self.fc2 = torch.nn.Linear(1024**2, numclasses)
		self.fc3 = torch.nn.Linear(2048**2, numclasses)
		self.fc = [self.fc1, self.fc2, self.fc3]

		# 冻结骨干网络
		for param in self.backbone.parameters():
			param.requires_grad = False
		# 初始化 fc 层
		for layer in self.fc:
			torch.nn.init.kaiming_normal(layer.weight.data)
			if layer.bias is not None:
				torch.nn.init.constant(layer.bias.data, val=0)

		if 1 <= self.phi <= 4:
			self.feat1_att = attention_block[self.phi - 1](512)
			self.feat2_att = attention_block[self.phi - 1](1024)
			self.feat3_att = attention_block[self.phi - 1](2048)

	def forward(self, x):
		N = x.size()[0]
		assert x.size() == (N, 3, 224, 224)
		feat1, feat2, feat3 = self.backbone(x)
		if 1 <= self.phi <= 4:
			feat1 = self.feat1_att(feat1)
			assert feat1.size() == (N, 512, 28, 28)
			feat2 = self.feat2_att(feat2)
			assert feat2.size() == (N, 1024, 14, 14)
			feat3 = self.feat3_att(feat3)
			assert feat3.size() == (N, 2048, 7, 7)

		feat1 = feat1.view(N, 512, 28**2)
		feat2 = feat2.view(N, 1024, 14**2)
		feat3 = feat3.view(N, 2048, 7**2)

		feat1 = torch.bmm(feat1, torch.transpose(feat1, 1, 2)) / (28**2)
		feat2 = torch.bmm(feat2, torch.transpose(feat2, 1, 2)) / (14**2)
		feat3 = torch.bmm(feat3, torch.transpose(feat3, 1, 2)) / (7**2)
		assert feat1.size() == (N, 512, 512)
		assert feat2.size() == (N, 1024, 1024)
		assert feat3.size() == (N, 2048, 2048)

		feat1 = feat1.view(N, 512**2)
		feat2 = feat2.view(N, 1024**2)
		feat3 = feat3.view(N, 2048**2)

		feat1 = torch.sqrt(feat1 + 1e-5)
		feat2 = torch.sqrt(feat2 + 1e-5)
		feat3 = torch.sqrt(feat3 + 1e-5)

		feat1 = torch.nn.functional.normalize(feat1)
		feat2 = torch.nn.functional.normalize(feat2)
		feat3 = torch.nn.functional.normalize(feat3)

		out1 = self.fc1(feat1)
		out2 = self.fc2(feat2)
		out3 = self.fc3(feat3)

		return out1, out2, out3


if __name__ == '__main__':
	inputs = torch.rand((1, 3, 224, 224))
	model = ResNetWithAttenBili()
	outputs = model(inputs)
	for output in outputs:
		print(output.shape)




