# coding=utf-8

"""
Author: xiezhenqing
Email: 441798804@qq.com
date: 2022/7/28 12:49
desc: 骨干网络可选：
		 MobileNetV2、MobileNetV3Large、MobileNetV3Small、ShuffleNetV2_x1_0、
	  输出特征层大小:
	  MobileNetV2、MobileNetV3Large、GhostNet 为 (N, 160, 7, 7)，
	  MobileNetV3Small 为 (N, 96, 7, 7)，
	  ShuffleNetV2_x1_0 为 (N, 464, 7, 7)
	  ResNet50 为 (N, 2048, 7, 7)
	  为最后一个stage中三个block分别输出，
	  三个输出后接注意力机制模块 可选（SE、CBAM、ECA、CA），
	  后经映射层变换通道数量到 proj_k 大小(原论文为8192)，考虑运行速度进行缩减，进行尝试其它超参数，
	  三个变换后的特征层两两相乘，取平均池化后变换成大小为 [batch_size, -1] 的向量，再标准化后拼接输出softmax结果
"""

import torch
import torch.nn as nn

from .mobilenet_V2 import mobilenet_v2
from .mobilenet_V3 import mobilenet_v3_large, mobilenet_v3_small
from .shufflenet import shufflenet_v2_x1_0
from .ghostnet import ghostnet
from .resnet import resnet50
from .attention import se_block, cbam_block, eca_block, CA_Block


attention_block = [se_block, cbam_block, eca_block, CA_Block]


class MobileNetV2(nn.Module):
	def __init__(self, pretrained = False):
		super(MobileNetV2, self).__init__()
		model = mobilenet_v2(pretrained=pretrained)
		del model.classifier
		del model.avgpool
		del model.features[18]
		del model.features[17]
		self.model = model

	def forward(self, x):
		# for i in range(len(self.model.features)):
		# 	x = self.model.features[i](x)
		# 	print(x.shape)
		out3 = self.model.features[:15](x)          # (1, 160, 7, 7)
		out4 = self.model.features[15:16](out3)     # (1, 160, 7, 7)
		out5 = self.model.features[16:17](out4)     # (1, 160, 7, 7)
		return out3, out4, out5


class MobileNetV3Large(nn.Module):
	def __init__(self, pretrained=False):
		super(MobileNetV3Large, self).__init__()
		model = mobilenet_v3_large(pretrained=pretrained)
		del model.classifier
		del model.avgpool
		del model.features[16]
		self.model = model

	def forward(self, x):
		# for i in range(len(self.model.features)):
		# 	x = self.model.features[i](x)
		# 	print(x.shape)
		out3 = self.model.features[:14](x)            # (1, 160, 7, 7)
		out4 = self.model.features[14:15](out3)       # (1, 160, 7, 7)
		out5 = self.model.features[15:16](out4)       # (1, 160, 7, 7)
		return out3, out4, out5


class MobileNetV3Small(nn.Module):
	def __init__(self, pretrained=False):
		super(MobileNetV3Small, self).__init__()
		model = mobilenet_v3_small(pretrained=pretrained)
		del model.classifier
		del model.avgpool
		del model.features[12]
		self.model = model

	def forward(self, x):
		# for i in range(len(self.model.features)):
		# 	x = self.model.features[i](x)
		# 	print(x.shape)
		out3 = self.model.features[:10](x)            # (1, 96, 7, 7)
		out4 = self.model.features[10:11](out3)       # (1, 96, 7, 7)
		out5 = self.model.features[11:12](out4)       # (1, 96, 7, 7)
		return out3, out4, out5


class ShuffleNetV2_x1_0(nn.Module):
	def __init__(self, pretrained=False):
		super(ShuffleNetV2_x1_0, self).__init__()
		model = shufflenet_v2_x1_0(pretrained=pretrained)
		del model.fc
		del model.conv5
		self.model = model

	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.maxpool(x)
		x = self.model.stage2(x)
		x = self.model.stage3(x)
		x = self.model.stage4[0](x)
		out3 = self.model.stage4[1](x)
		out4 = self.model.stage4[2](out3)
		out5 = self.model.stage4[3](out4)

		return out3, out4, out5


class GhostNet(nn.Module):
	def __init__(self, pretrained=False):
		super(GhostNet, self).__init__()
		model = ghostnet(pretrained=pretrained)
		del model.global_pool
		del model.conv_head
		del model.act2
		del model.classifier
		del model.blocks[9]
		self.model = model

	def forward(self, x):
		x = self.model.conv_stem(x)
		x = self.model.bn1(x)
		x = self.model.act1(x)
		x = self.model.blocks[:8](x)
		x = self.model.blocks[8][0](x)
		out3 = self.model.blocks[8][1](x)
		out4 = self.model.blocks[8][2](out3)
		out5 = self.model.blocks[8][3](out4)

		return out3, out4, out5


class ResNet50(nn.Module):
	def __init__(self, pretrained=False):
		super(ResNet50, self).__init__()
		model = resnet50(pretrained=pretrained)
		del model.fc
		del model.avgpool
		self.model = model

	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)

		out3 = self.model.layer4[0](x)
		out4 = self.model.layer4[1](out3)
		out5 = self.model.layer4[2](out4)

		return out3, out4, out5


class HBPNet(nn.Module):
	def __init__(self,
	             backbone: str,             # 选择骨干网络
	             phi: int = 1,              # 选择注意力机制模块 (1 -> 4)
	             pretrained: bool = False,  # 加载预训练权重
	             proj_k:int = 8192,         # 将输出通道数映射到相应通道数目
	             num_classes:int = 1000     # 类别数
	             ):
		super(HBPNet, self).__init__()
		self.phi = phi
		self.backbone = backbone

		if self.phi > 4 or self.phi < 1:
			raise ValueError('phi定义超过相应范围')

		if self.backbone == 'MobileNetV2':
			self.backbone = MobileNetV2(pretrained=pretrained)
			out_filter = 160
		elif self.backbone == 'MobileNetV3Large':
			self.backbone = MobileNetV3Large(pretrained=pretrained)
			out_filter = 160
		elif self.backbone == 'MobileNetV3Small':
			self.backbone = MobileNetV3Small(pretrained=pretrained)
			out_filter = 96
		elif self.backbone == 'ShuffleNetV2_x1_0':
			self.backbone = ShuffleNetV2_x1_0(pretrained=pretrained)
			out_filter = 464
		elif self.backbone == 'GhostNet':
			self.backbone = GhostNet(pretrained=pretrained)
			out_filter = 160
		elif self.backbone == 'ResNet50':
			self.backbone = ResNet50(pretrained=pretrained)
			out_filter = 2048
		else:
			raise ValueError('没有定义该骨干网络')

		self.feat1_att = attention_block[self.phi - 1](out_filter)
		self.feat2_att = attention_block[self.phi - 1](out_filter)
		self.feat3_att = attention_block[self.phi - 1](out_filter)
		self.proj1 = nn.Conv2d(out_filter, proj_k, kernel_size=1, stride=1)
		self.proj2 = nn.Conv2d(out_filter, proj_k, kernel_size=1, stride=1)
		self.proj3 = nn.Conv2d(out_filter, proj_k, kernel_size=1, stride=1)
		self.fc_concat = nn.Linear(proj_k*3, num_classes)
		self.softmax = nn.LogSoftmax(dim=1)
		self.avgpool = nn.AvgPool2d(kernel_size=7)

	def forward(self, x):
		batch_size = x.size(0)
		feat1, feat2, feat3 = self.backbone(x)
		# print('feat1.shape = ', feat1.shape)
		# print('feat2.shape = ', feat2.shape)
		# print('feat3.shape = ', feat3.shape)

		feat1 = self.feat1_att(feat1)
		feat2 = self.feat2_att(feat2)
		feat3 = self.feat3_att(feat3)

		feat1 = self.proj1(feat1)
		feat2 = self.proj2(feat2)
		feat3 = self.proj3(feat3)

		inter1 = feat1 * feat2
		inter2 = feat1 * feat3
		inter3 = feat2 * feat3

		inter1 = self.avgpool(inter1).view(batch_size, -1)
		inter2 = self.avgpool(inter2).view(batch_size, -1)
		inter3 = self.avgpool(inter3).view(batch_size, -1)

		result1 = nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
		result2 = nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
		result3 = nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))

		result = torch.cat((result1, result2, result3), 1)
		result = self.fc_concat(result)

		return self.softmax(result)

	def freeze_backbone(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False

	def Unfreeze_backbone(self):
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True


if __name__ == '__main__':
	# # from fvcore.nn import FlopCountAnalysis, parameter_count_table
	# # from torchsummary import summary
	# # model = HBPNet(backbone='ResNet50', phi=1, num_classes=6)
	# # inputs = torch.rand(1, 3, 224, 224)
	# # flops = FlopCountAnalysis(model, inputs)
	# # print('FLOPs: ', flops.total())
	# # print(parameter_count_table(model))
	# # summary(model, (3, 224, 224), 8)
	# # print(model)
	#
	# model = MobileNetV3Small().eval()
	# inputs = torch.randn(8, 3, 224, 224)
	# outputs = model(inputs)
	# for output in outputs:
	# 	print(output.shape)

	model = HBPNet('ResNet50')
	model.freeze_backbone()

	for k, v in model.named_parameters():
		print('{}: {}'.format(k, v.requires_grad))
