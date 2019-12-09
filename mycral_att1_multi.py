import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torchsummary import summary
import cv2
import numpy as np
import math

class MyCRAL(nn.Module):

	def __init__(self):
		super(MyCRAL,self).__init__()

		densenet_model = models.densenet121(pretrained=True)
		features = list(densenet_model.children())[:-1]
		self.features = nn.Sequential(*features)

		#freezing the densenet model weights
		for param in densenet_model.parameters():
			param.requires_grad = True

		self.att1 = nn.Sequential(OrderedDict([
						  ('att_conv_1',nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)),
						  ('relu_1', nn.ReLU()),
						  ('att_conv_2',nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)),
						  ('relu_2', nn.ReLU()),
						  ('att_conv_3',nn.Conv2d(in_channels=1024, out_channels=14, kernel_size=1, stride=1))
						  ]))

		self.upsample = nn.Upsample(size=(10, 10))
		self.pooling = nn.AvgPool2d(kernel_size=10, stride=1)
		self.output = nn.Linear(in_features = 14336,out_features=42)

	def forward(self, x):

		feature_map = self.features(x)
		feature_map = F.relu(feature_map)
		
		attention_map  = self.att1(feature_map)
		attention_map = self.upsample(attention_map)

		attention_map  = F.sigmoid(attention_map)
		ones_tensor = torch.FloatTensor(np.ones(attention_map.shape)).cuda()
		attention_map = ones_tensor + attention_map

		attn_feature_hadamard_sum = torch.einsum("ijkm,ilkm->ijlkm", (feature_map, attention_map))
		attn_feature_hadamard_sum = attn_feature_hadamard_sum.reshape(feature_map.shape[0], -1, feature_map.shape[2], feature_map.shape[3])

		residual_attn_features = F.relu(attn_feature_hadamard_sum)	
		residual_attn_pooled = self.pooling(residual_attn_features)	
		x = residual_attn_pooled.view(residual_attn_pooled.size(0), -1)

		#print('Output')
		out = self.output(x)		
		out = out.reshape([len(out), 3, 14])
		out = F.softmax(out, dim =1)

		return out