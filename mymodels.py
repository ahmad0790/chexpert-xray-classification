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

#Credit: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]

	return rmin, rmax, cmin, cmax

#ResNet152
def MyResNet():

	resnet_model = models.resnet50(pretrained=True)

	INPUT_NODES = 2048
	OUTPUT_NODES = 14

	#freezing the densenet model weights
	for param in resnet_model.parameters():
		param.requires_grad = True

	#changing the classification layer
	resnet_model.fc = nn.Linear(INPUT_NODES, OUTPUT_NODES)

	for param in resnet_model.fc.parameters():
		param.requires_grad = True

	return resnet_model


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

		self.att2 =  nn.Sequential(OrderedDict([
						  ('att_pool_1',nn.MaxPool2d(kernel_size=3, stride=1)),
						  ('att_pool_2',nn.MaxPool2d(kernel_size=3, stride=1)),
						  ('att_pool_3',nn.MaxPool2d(kernel_size=2, stride=1)),
						  ('upsample_1',nn.Upsample(size=(6, 6))),
						  ('upsample_2',nn.Upsample(size=(8, 8))),
						  ('upsample_3',nn.Upsample(size=(10, 10))),
						  ('att_conv_1',nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1)),
						  ('att_conv_2',nn.Conv2d(in_channels=1024, out_channels=14, kernel_size=1, stride=1))
						  ]))

		self.upsample = nn.Upsample(size=(10, 10))
		self.pooling = nn.AvgPool2d(kernel_size=10, stride=1)
		self.output = nn.Linear(in_features = 14336,out_features=14)

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

		out = self.output(x)

		return out

#AGCNN
def get_local_activated_patch(image, feature_map):
	
	#print('Image Patching')
	feature_map = np.abs(feature_map.data.cpu().numpy())
	image = image.data.cpu().numpy()
	img_size = 320
	
	#16*1024*7*7
	batch_size, c, h, w = feature_map.shape
	local_patches = []

	for i in range(0, batch_size):
		
		feature_map_batch_example = feature_map[i]
		Hg = feature_map_batch_example.max(axis=0)

		threshold = np.percentile(Hg, 70)

		Hg = cv2.resize(Hg, (img_size, img_size))

		Hg[Hg<=threshold] = 0
		Hg[Hg>threshold] = 1

		min_y, max_y, min_x, max_x = bbox(Hg)

		Ic = image[i, :, min_y:max_y, min_x:max_x]

		local_ptch = np.swapaxes(Ic,0,2)
		local_ptch = np.swapaxes(local_ptch,0,1)

		Ic_resized = cv2.resize(local_ptch, (img_size, img_size))

		Ic_resized = np.swapaxes(Ic_resized,0,2)
		Ic_resized = np.swapaxes(Ic_resized,1,2)

		local_patches.append(Ic_resized)


	local_patches = torch.FloatTensor(local_patches).cuda()

	return local_patches


class MyAGCNN(nn.Module):

	def __init__(self):
		super(MyAGCNN,self).__init__()

		densenet_model = models.densenet121(pretrained=True)
		features = list(densenet_model.children())[:-1] #2 if resnet
		self.features = nn.Sequential(*features)

		#freezing the densenet model weights
		for param in densenet_model.parameters():
			param.requires_grad = True

		#self.pooling = nn.MaxPool2d(kernel_size=7, stride=1)
		self.pooling = nn.MaxPool2d(kernel_size=10, stride=1) #if using 320 image size
		self.output = nn.Linear(in_features = 1024,out_features=14)
		self.Sigmoid = nn.Sigmoid()


	def forward(self, x):
		
		feature_map = self.features(x)
		feature_map = F.relu(feature_map)
		out_pooling = self.pooling(feature_map)
		x = out_pooling.view(feature_map.size(0), -1)
		x = self.output(x)
		out = self.Sigmoid(x)
		
		return out, feature_map, out_pooling


class MyAGCNN_CRAL(nn.Module):

	def __init__(self):
		super(MyAGCNN_CRAL,self).__init__()

		densenet_model = models.densenet121(pretrained=True)
		features = list(densenet_model.children())[:-1] #2 if resnet
		self.features = nn.Sequential(*features)

		for param in densenet_model.parameters():
			param.requires_grad = True

		#self.pooling = nn.MaxPool2d(kernel_size=10, stride=1) #if using 320 image size
		self.output = nn.Linear(in_features = 14336,out_features=42)
		self.upsample = nn.Upsample(size=(10, 10))
		self.pooling = nn.AvgPool2d(kernel_size=10, stride=1)
		self.Sigmoid = nn.Sigmoid()

		self.att1 = nn.Sequential(OrderedDict([
				  ('att_conv_1',nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)),
				  ('relu_1', nn.ReLU()),
				  ('att_conv_2',nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)),
				  ('relu_2', nn.ReLU()),
				  ('att_conv_3',nn.Conv2d(in_channels=1024, out_channels=14, kernel_size=1, stride=1))
				  ]))


	def forward(self, x):
		
		feature_map = self.features(x)
		feature_map = F.relu(feature_map)
		attention_map  = self.att1(feature_map)
		attention_map  = F.sigmoid(attention_map)
		attention_map = self.upsample(attention_map)
		ones_tensor = torch.FloatTensor(np.ones(attention_map.shape)).cuda()
		attention_map = ones_tensor + attention_map
		attn_feature_hadamard_sum = torch.einsum("ijkm,ilkm->ijlkm", (feature_map, attention_map))
		attn_feature_hadamard_sum = attn_feature_hadamard_sum.reshape(feature_map.shape[0], -1, feature_map.shape[2], feature_map.shape[3])
		residual_attn_features = F.relu(attn_feature_hadamard_sum)	
		out_pooling = self.pooling(residual_attn_features)	
		x = out_pooling.view(out_pooling.size(0), -1)
		out = self.output(x)		
		out = out.reshape([len(out), 3, 14])
		out = F.softmax(out, dim =1)

		return out, feature_map, out_pooling


class AGCNN_Fusion(nn.Module):
	def __init__(self):
		super(AGCNN_Fusion, self).__init__()
		self.fc = nn.Linear(2048, 14)
		self.Sigmoid = nn.Sigmoid()

	def forward(self, global_pool, local_pool):

		fusion = torch.cat((global_pool,local_pool), 1).cuda()
		fusion = fusion.reshape([len(fusion), 2048])
		x = self.fc(fusion)
		x = self.Sigmoid(x)
		return x

class AGCNN_Fusion_Cral(nn.Module):
	def __init__(self):
		super(AGCNN_Fusion_Cral, self).__init__()
		self.fc = nn.Linear(14336*2, 42)
		self.Sigmoid = nn.Sigmoid()

	def forward(self, global_pool, local_pool):
		fusion = torch.cat((global_pool,local_pool), 1).cuda()
		fusion = fusion.reshape([len(fusion), 14336*2])
		x = self.fc(fusion)
		x = x.reshape([len(x), 3, 14])
		x = F.softmax(x, dim =1)
		return x


class MyDenseNetMultiClass(nn.Module):

	def __init__(self):
		super(MyDenseNetMultiClass,self).__init__()

		densenet_model = models.densenet121(pretrained=True)
		features = list(densenet_model.children())[:-1]
		self.features = nn.Sequential(*features)
		self.output = nn.Linear(in_features = 102400,out_features=42)

		for param in densenet_model.parameters():
			param.requires_grad = True

	def forward(self, x):

		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.output(x)
		x = x.reshape([len(x), 3, 14])
		x = F.softmax(x, dim =1)

		return x

def MyDenseNet():

	densenet_model = models.densenet121(pretrained=True)

	#not freezing the densenet model weights
	for param in densenet_model.parameters():
		param.requires_grad = True

	#changing the classification layer
	densenet_model.classifier = nn.Linear(1024, 14)

	for param in densenet_model.classifier.parameters():
		param.requires_grad = True

	return densenet_model