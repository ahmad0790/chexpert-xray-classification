#Load and convert to Pytorch Dataset
import os
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image

class ImagesDataset(Dataset):
	def __init__(self, root_dir, labels_file, labels_type, transform = None, multiclass=False):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		##read labels

		'''
		self.labels = pd.read_csv(os.path.join(root_dir, labels_file))
		self.labels = np.array(labels.iloc[:,5:])
		
		if labels_type == 'Ones':
			self.labels[self.labels==-1] = 1
		'''

		images_list = []
		labels_list = []
		
		with open(os.path.join(root_dir, labels_file), "r") as f:
			rows_read=0

			for line in f:

				if rows_read > 0:
				
					cols = line.strip('\n').split(',')
					image_name= cols[0]
					#print(image_name)
					
					labels = cols[5:]

					#for multiclass
					if multiclass == True:
						labels_copy = np.zeros((3,14))
					else:
						labels_copy = np.zeros(14)

					for i in range(0, len(labels)):
						
						##Ones Classification
						if multiclass == False:

							if labels[i]=='-1.0':
								labels_copy[i] = 1.0
							
							elif labels[i] == '1.0':
								labels_copy[i] = 1.0
						
						#multiclass classification
						elif multiclass == True:	
							
							if labels[i]=='-1.0':
								#labels_uncertain[i] = 1.0
								labels_copy[0,i] = 1.0
							elif labels[i] == '1.0':
								#labels_ones[i] = 1.0
								labels_copy[1,i] = 1.0

							elif labels[i] in ('0.0',''):
								#labels_zero[i] = 1.0
								labels_copy[2,i] = 1.0

					labels_copied = np.copy(labels_copy)

					#print(labels_copy)
					if multiclass == True:
						#labels_list.append([labels_zero, labels_ones, labels_uncertain])
						labels_list.append([labels_copied[2,:], labels_copied[1,:], labels_copied[0,:]])

					else:
						labels_list.append(labels_copied)

					images_list.append(image_name)

				rows_read += 1

				#print(labels_list)

		self.images = images_list
		self.labels_list = labels_list
		self.transform = transform

	def __len__(self):
		return len(self.labels_list)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader

		image_name = self.images[index]
		#print(image_name)
		image = Image.open(image_name).convert('RGB')
		label = self.labels_list[index]
		if self.transform is not None:
			image = self.transform(image)
		return image, torch.FloatTensor(label)
