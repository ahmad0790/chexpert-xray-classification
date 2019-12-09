import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from mymodels import get_local_activated_patch
import cv2
import os
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from read_images_dataset import ImagesDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
from plots import plot_learning_curves, plot_confusion_matrix
from torchsummary import summary

warnings.filterwarnings('ignore')

#image net mean values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

image_transformations = transforms.Compose([
   transforms.Resize((320,320)),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   normalize,
])

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def compute_class_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():

		batch_size = target.size(0)
		correct = output.eq(target).sum()

		return correct * 100.0 / batch_size

#modify this function to reflect multilabel accuracy
def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	accuracy_list = []
	with torch.no_grad():

		batch_size = target.size(0)
		prediction = output.data
		prediction[prediction>=0.5] = 1
		prediction[prediction<0.5] = 0

		for i in range(0,14):
			accuracy_list.append(compute_class_accuracy(prediction[:,i], target[:,i]))

		return sum(accuracy_list)/14.00

##for training the AG-CNN model
def train_agcnn(global_model, local_model, fusion_model, device, data_loader, criterion
	, optimizer_global, optimizer_local, optimizer_fusion, epoch, print_freq=10):
	
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	global_losses = AverageMeter()
	local_losses = AverageMeter()
	fusion_losses = AverageMeter()
	accuracy = AverageMeter()

	global_model.train()
	local_model.train()
	fusion_model.train()

	end = time.time()
	j = 0
	
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time

		if j < np.float('inf'):

			data_time.update(time.time() - end)

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			
			target = target.to(device)

			optimizer_global.zero_grad()
			optimizer_local.zero_grad()
			optimizer_fusion.zero_grad()

			global_output, global_fm, global_pool = global_model(input)
			local_image_patch = get_local_activated_patch(input, global_fm)
			local_output, local_fm, local_pool = local_model(input)
			fusion_output = fusion_model(global_pool, local_pool)

			global_loss = criterion(global_output, target)
			local_loss = criterion(local_output, target)
			fusion_loss = criterion(fusion_output, target)

			loss = global_loss + local_loss + fusion_loss

			loss.backward()
			optimizer_global.step()
			optimizer_local.step()
			optimizer_fusion.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			global_losses.update(global_loss.item(), target.size(0))
			local_losses.update(local_loss.item(), target.size(0))
			fusion_losses.update(fusion_loss.item(), target.size(0))

			accuracy.update(compute_batch_accuracy(fusion_output, target).item(), target.size(0))

			assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

			if i % print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Global_Loss {loss_global.val:.4f} ({loss_global.avg:.4f})\t'
					  'Local_Loss {loss_local.val:.4f} ({loss_local.avg:.4f})\t'
					  'Fusion_Loss {loss_fusion.val:.4f} ({loss_fusion.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					epoch, i, len(data_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, loss_global = global_losses
					, loss_local = local_losses, loss_fusion = fusion_losses,   acc=accuracy))
			j+=1

		else:
			break


	return losses.avg, accuracy.avg, global_losses.avg, local_losses.avg, fusion_losses.avg

##for training the AG-CNN 3 Class Classification model
def train_agcnn_multi(global_model, local_model, fusion_model, device, data_loader, criterion
	, optimizer_global, optimizer_local, optimizer_fusion, epoch, print_freq=10):
	
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	global_losses = AverageMeter()
	local_losses = AverageMeter()
	fusion_losses = AverageMeter()
	accuracy = AverageMeter()

	global_model.train()
	local_model.train()
	fusion_model.train()

	end = time.time()
	j = 0
	
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time

		if j < np.float('inf'):

			data_time.update(time.time() - end)

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			
			target = target.to(device)

			optimizer_global.zero_grad()
			optimizer_local.zero_grad()
			optimizer_fusion.zero_grad()

			global_output, global_fm, global_pool = global_model(input)
			local_image_patch = get_local_activated_patch(input, global_fm)
			local_output, local_fm, local_pool = local_model(input)
			fusion_output = fusion_model(global_pool, local_pool)

			global_loss = criterion(global_output, target)
			local_loss = criterion(local_output, target)
			fusion_loss = criterion(fusion_output, target)

			loss = global_loss + local_loss + fusion_loss

			loss.backward()
			optimizer_global.step()
			optimizer_local.step()
			optimizer_fusion.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			global_losses.update(global_loss.item(), target.size(0))
			local_losses.update(local_loss.item(), target.size(0))
			fusion_losses.update(fusion_loss.item(), target.size(0))

			accuracy.update(compute_batch_accuracy(fusion_output[:,1,:], target[:,1,:]).item(), target.size(0))

			assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

			if i % print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Global_Loss {loss_global.val:.4f} ({loss_global.avg:.4f})\t'
					  'Local_Loss {loss_local.val:.4f} ({loss_local.avg:.4f})\t'
					  'Fusion_Loss {loss_fusion.val:.4f} ({loss_fusion.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					epoch, i, len(data_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, loss_global = global_losses
					, loss_local = local_losses, loss_fusion = fusion_losses,   acc=accuracy))
			j+=1

		else:
			break


	return losses.avg, accuracy.avg, global_losses.avg, local_losses.avg, fusion_losses.avg


##for training the DenseNet, ResNet and CRAL models
def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10, multiclass=False):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()
	auc = AverageMeter()

	model.train()
	end = time.time()
	j = 0

	for i, (input, target) in enumerate(data_loader):

		if j < np.float('inf'):
		#if j < 10:

			data_time.update(time.time() - end)

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			
			target = target.to(device)

			optimizer.zero_grad()

			output = model(input)
			loss = criterion(output, target)

			loss.backward()
			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			
			if multiclass == True:
				accuracy.update(compute_batch_accuracy(output[:,1,:], target[:,1,:]).item(), target.size(0))
			else:
				accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

			assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

			if i % print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					epoch, i, len(data_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, acc=accuracy))
			j+=1

		else:
			break

	return losses.avg, accuracy.avg


##for evaluating the densenet, resnet, and CRAL models
def evaluate(model, device, data_loader, criterion, print_freq=10, multiclass = False):
	
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()
	auc = AverageMeter()

	results = []
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			
			target = target.to(device)
				
			output = model(input)
			loss = criterion(output, target)

			losses.update(loss.item(), target.size(0))
			
			if multiclass == True:
				accuracy.update(compute_batch_accuracy(output[:,1,:], target[:,1,:]).item(), target.size(0))
				y_true = target[:,1,:].detach().to('cpu').numpy().tolist()
				y_pred = output[:,1,:].detach().to('cpu').max(1)[1].numpy().tolist()

			else:
				accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
				y_true = target.detach().to('cpu').numpy().tolist()
				y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
				
			results.extend(list(zip(y_true, y_pred)))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results


##for evaluating the AG-CNN model
def evaluate_agcnn(fusion_model, global_model, local_model, device, data_loader, criterion, print_freq=10, multiclass=False):
	
	batch_time = AverageMeter()
	losses = AverageMeter()
	global_losses = AverageMeter()
	local_losses = AverageMeter()
	fusion_losses = AverageMeter()
	accuracy = AverageMeter()

	results = []
	global_model.eval()
	local_model.eval()
	fusion_model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			
			target = target.to(device)

			global_output, global_feature_map, global_pooling  = global_model(input)
			local_image_crop = get_local_activated_patch(input, global_feature_map)	
			local_output, local_feature_map, local_pooling  = local_model(local_image_crop)
			fusion_output  = fusion_model(global_pooling, local_pooling)

			global_loss = criterion(global_output, target)
			local_loss = criterion(local_output, target)
			fusion_loss = criterion(fusion_output, target)
			loss = global_loss + local_loss + fusion_loss

			losses.update(loss.item(), target.size(0))
			global_losses.update(global_loss.item(), target.size(0))
			local_losses.update(local_loss.item(), target.size(0))
			fusion_losses.update(fusion_loss.item(), target.size(0))

			if multiclass == True:
				accuracy.update(compute_batch_accuracy(fusion_output[:,1,:], target[:,1,:]).item(), target.size(0))
			else:
				accuracy.update(compute_batch_accuracy(fusion_output, target).item(), target.size(0))
	
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = fusion_output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Global_Loss {loss_global.val:.4f} ({loss_global.avg:.4f})\t'
					  'Local_Loss {loss_local.val:.4f} ({loss_local.avg:.4f})\t'
					  'Fusion_Loss {loss_fusion.val:.4f} ({loss_fusion.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, loss_global = global_losses, loss_local = local_losses, loss_fusion = fusion_losses, acc=accuracy))

	return losses.avg, global_losses.avg, local_losses.avg, fusion_losses.avg, accuracy.avg, results


##for predicting disease for the DEnsenet, resnet and CRAL models
def predict_disease(model, device, data_loader, multiclass=False):
	
	model.eval()
	probas = np.empty((0,14),dtype=np.float32)
	labels = np.empty((0,14),dtype=np.float32)

	with torch.no_grad():
	
		for i, (batch, target) in enumerate(data_loader):

			if isinstance(batch, tuple):
				batch = tuple([e.to(device) if type(e) == torch.Tensor else e for e in batch])
			else:
				batch = batch.to(device)
	
			target = target.to(device)

			output = model(batch)
			prediction = F.sigmoid(output)
			target = target.cpu().numpy()

			if multiclass == True:
				prediction = prediction[:,0:2,:]
				prediction = F.softmax(prediction, dim=1)
				prediction = prediction[:,1,:]
				target = target[:,1,:]

			prediction = prediction.cpu().numpy()
			probas = np.concatenate((probas, prediction), axis=0)
			labels = np.concatenate((labels, target), axis=0)

	return probas, labels

##for predicting disease on the AG-CNN model
def predict_disease_agcnn(global_model, local_model, fusion_model, device, data_loader, branch='global', multiclass=False):
	
	global_model.eval()
	local_model.eval()
	fusion_model.eval()

	probas = np.empty((0,14),dtype=np.float32)
	labels = np.empty((0,14),dtype=np.float32)

	with torch.no_grad():
	
		for i, (batch, target) in enumerate(data_loader):

			if isinstance(batch, tuple):
				batch = tuple([e.to(device) if type(e) == torch.Tensor else e for e in batch])
			else:
				batch = batch.to(device)
	
			target = target.to(device)

			global_output, global_feature_map, global_pooling  = global_model(batch)
			local_image_crop = get_local_activated_patch(batch, global_feature_map)	
			local_output, local_feature_map, local_pooling  = local_model(local_image_crop)
			fusion_output  = fusion_model(global_pooling, local_pooling)

			if branch == 'fusion':
				output = fusion_output
			elif branch == 'global':
				output = global_output
			elif branch == 'local':
				output = local_output

			prediction = output.data
			target = target.cpu().numpy()

			if multiclass == True:
				prediction = prediction[:,0:2,:]
				prediction = F.softmax(prediction, dim=1)
				prediction = prediction[:,1,:]
				target = target[:,1,:]
			else:
				prediction = F.sigmoid(output)
			
			prediction = prediction.cpu().numpy()
			probas = np.concatenate((probas, prediction), axis=0)
			labels = np.concatenate((labels, target), axis=0)

	return probas, labels