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
from skimage.measure import label

warnings.filterwarnings('ignore')

#image net mean values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

image_transformations = transforms.Compose([
   transforms.Resize((320,320)),
   #transforms.CenterCrop(224),
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
		#_, pred = output.max(1)
		correct = output.eq(target).sum()

		return correct * 100.0 / batch_size

#modify this function to reflect multilabel accuracy
def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	accuracy_list = []
	with torch.no_grad():

		batch_size = target.size(0)
		#print(batch_size)
		prediction = output.data
		#print(prediction)
		#target = target.data
		prediction[prediction>=0.5] = 1
		prediction[prediction<0.5] = 0

		#print(prediction[:,1])

		for i in range(0,14):
			accuracy_list.append(compute_class_accuracy(prediction[:,i], target[:,i]))

		return sum(accuracy_list)/14.00


'''
def get_local_activated_patch(ori_image, fm_cuda):
    # fm => mask =>(+ ori-img) => crop = patchs
    feature_conv = fm_cuda.data.cpu().numpy()
    size_upsample = (224, 224) 
    bz, nc, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().cuda()

    for i in range(0, bz):
        feature = feature_conv[i]

        cam = feature.reshape((nc, h*w))
        #print(cam.shape)
        #print(cam)

        cam = cam.sum(axis=0)
        cam = cam.reshape(h,w)

        #min max normalization
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        #print(cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        
        # to ori image 
        image = ori_image[i].cpu().numpy().reshape(224,224,3)
        image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh,minw:maxw,:] * 256 # because image was normalized before
        image_crop = image_transformations(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3,224,224).unsqueeze(0).cuda())

        patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    return patchs_cuda


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 
'''


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
		#if j < 10:

			data_time.update(time.time() - end)

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			
			target = target.to(device)

			optimizer_global.zero_grad()
			optimizer_local.zero_grad()
			optimizer_fusion.zero_grad()

			#print(input.shape)

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
		# measure data loading time
		#print(input.shape)
		#print(target.shape)

		if j < np.float('inf'):
		#if j < 10:

			data_time.update(time.time() - end)

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			
			target = target.to(device)

			optimizer.zero_grad()

			if multiclass == False:
				output = model(input)
				loss = criterion(output, target)

				loss.backward()
				optimizer.step()

				# measure elapsed time
				batch_time.update(time.time() - end)
				end = time.time()

				losses.update(loss.item(), target.size(0))
				#accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
				accuracy.update(compute_batch_accuracy(output[:,1,:], target[:,1,:]).item(), target.size(0))

			elif multiclass == True:

				output1, output2, output3 = model(input)

				loss1 = criterion(output1, target[:,0,:])
				loss2 = criterion(output2, target[:,1,:])
				loss3 = criterion(output3, target[:,2,:])

				loss = loss1 + loss2 + loss3

				loss.backward()
				optimizer.step()

				# measure elapsed time
				batch_time.update(time.time() - end)
				end = time.time()

				losses.update(loss.item(), target[:,0,:].size(0))
				
				acc1 = compute_batch_accuracy(output1, target[:,0,:])
				acc2 = compute_batch_accuracy(output2, target[:,1,:])
				acc3 = compute_batch_accuracy(output3, target[:,2,:])
				#print(acc1, acc2, acc3)
				weighted_accuracy = (acc1+acc2+acc3)/3.0000
				accuracy.update(weighted_accuracy, target[:,0,:].size(0))
			
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

			if multiclass == False:
				
				output = model(input)
				loss = criterion(output, target)

				losses.update(loss.item(), target.size(0))
				#accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
				accuracy.update(compute_batch_accuracy(output[:,1,:], target[:,1,:]).item(), target.size(0))

				y_true = target[:,1,:].detach().to('cpu').numpy().tolist()
				y_pred = output[:,1,:].detach().to('cpu').max(1)[1].numpy().tolist()

				#y_true = target.detach().to('cpu').numpy().tolist()
				#y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
				
				results.extend(list(zip(y_true, y_pred)))

			elif multiclass == True:
				
				output1, output2, output3 = model(input)

				loss1 = criterion(output1, target[:,0,:])
				loss2 = criterion(output2, target[:,1,:])
				loss3 = criterion(output3, target[:,2,:])

				loss = loss1 + loss2 + loss3

				losses.update(loss.item(), target[:,0,:].size(0))
			
				acc1 = compute_batch_accuracy(output1, target[:,0,:])
				acc2 = compute_batch_accuracy(output2, target[:,1,:])
				acc3 = compute_batch_accuracy(output3, target[:,2,:])
				#print(acc1, acc2, acc3)
				mean_accuracy = (acc1+acc2+acc3)/3.0000
				accuracy.update(mean_accuracy, target[:,0,:].size(0))

				y_true = target.detach().to('cpu').numpy().tolist()
				y_pred_1 = output1.detach().to('cpu').max(1)[1].numpy().tolist()
				y_pred_2 = output2.detach().to('cpu').max(1)[1].numpy().tolist()
				y_pred_3 = output3.detach().to('cpu').max(1)[1].numpy().tolist()

				results.extend(list(zip(y_true, [y_pred_1, y_pred_2, y_pred_3])))

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



def evaluate_agcnn(fusion_model, global_model, local_model, device, data_loader, criterion, print_freq=10):
	
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

			accuracy.update(compute_batch_accuracy(fusion_output[:,1,:], target[:,1,:]).item(), target.size(0))

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

def predict_disease_multiclass(model, device, data_loader):
	
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
			output1 = output[:,0,:] #0
			output2 = output[:,1,:]	#1
			output3 = output[:,2,:] #uncertain

			#print(output1 - output2)
			#print(output2 - output3)
			prediction = torch.cat([output1.data, output2.data], dim =0)
			#print(prediction)
			prediction = prediction.reshape([target.shape[0],2,14])
			#print(prediction[0,:,:])
			prediction = F.softmax(prediction, dim=1)
			#print(prediction.shape)
			#print(prediction[0,:,:])

			prediction = prediction[:,1,:].cpu().numpy() #keep only the positive prediction
			#print(prediction.shape)
			ones_target = target[:,1,:].cpu().numpy()
			probas = np.concatenate((probas, prediction), axis=0)
			labels = np.concatenate((labels, ones_target), axis=0)

	return probas, labels

'''
def predict_disease_agcnn(global_model, local_model, fusion_model, device, data_loader, branch):
	
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

			prediction = F.sigmoid(output)
			prediction = output.data.cpu().numpy()
			target = target.cpu().numpy()
			probas = np.concatenate((probas, prediction), axis=0)
			labels = np.concatenate((labels, target), axis=0)

	return probas, labels
'''

def predict_disease_agcnn(global_model, local_model, fusion_model, device, data_loader, branch, multiclass=True):
	
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