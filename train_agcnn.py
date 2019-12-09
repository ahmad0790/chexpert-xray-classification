import os
import re
import sys
import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from read_images_dataset import ImagesDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
from utils import train, evaluate, predict_disease, train_agcnn, evaluate_agcnn, predict_disease_agcnn
from plots import plot_learning_curves, plot_confusion_matrix
from mymodels import MyCNN, MyDenseNet, MyVGG16, MyResNet, MyAGCNN, MyDenseNetMultiClass, AGCNN_Fusion, get_local_activated_patch
from torchsummary import summary

# Some parameters
MODEL_TYPE = 'AG-CNN'
NUM_EPOCHS = 5
BATCH_SIZE = 16
USE_CUDA = True
NUM_WORKERS = 16
MULTICLASS_FLAG = False
save_file = 'ag-cnn-percentile.pth'
IMG_RESOLUTION = 320

# Path for saving model
PATH_OUTPUT = "output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

ROOT_DIR = "CheXpert-v1.0-small"
train_labels_file ='train.csv'
valid_labels_file ='valid.csv'

#image net mean values
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

image_transformations = transforms.Compose([
   transforms.Resize((IMG_RESOLUTION,IMG_RESOLUTION)),
   #transforms.CenterCrop(224),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   normalize,
])

train_dataset = ImagesDataset(ROOT_DIR, train_labels_file, 'Ones', image_transformations, multiclass=MULTICLASS_FLAG)
valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=MULTICLASS_FLAG)

print('Dataset Sizes')
print(len(train_dataset))
print(len(valid_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

if device == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

print('Training ' + str(MODEL_TYPE))

#model training
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()
global_model = MyAGCNN()
local_model = MyAGCNN()
fusion_model = AGCNN_Fusion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model.to(device)
local_model.to(device)
fusion_model.to(device)

criterion.to(device)
#print('Model Summary')
#print(summary(model, (3, 224, 224)))
optimizer_global = optim.Adam(global_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
optimizer_local = optim.Adam(local_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
optimizer_fusion = optim.Adam(fusion_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
#lr_global_decay = optim.lr_scheduler.StepLR(optimizer_global , step_size = 10, gamma = 1)
#lr_local_decay = optim.lr_scheduler.StepLR(optimizer_local , step_size = 10, gamma = 1)
#lr_fusion_decay = optim.lr_scheduler.StepLR(optimizer_fusion , step_size = 10, gamma = 1)

best_val_losses = 1000000.00
train_losses, train_accuracies, train_global_losses, train_local_losses, train_fusion_losses = [], [], [], [], []
valid_losses, valid_accuracies, valid_global_losses, valid_local_losses, valid_fusion_losses = [], [], [], [], []
 
for epoch in range(NUM_EPOCHS):

	#lr_global_decay.step()
	#lr_local_decay.step() 
	#lr_fusion_decay.step() 

	train_loss, train_accuracy, train_global_loss, train_local_loss, train_fusion_loss = train_agcnn(global_model, local_model, fusion_model, device, train_loader, criterion, optimizer_global, optimizer_local, optimizer_fusion, epoch)
	valid_loss, valid_global_loss, valid_local_loss, valid_fusion_loss, valid_accuracy, valid_results = evaluate_agcnn(fusion_model, global_model, local_model, device, valid_loader, criterion)
	
	print(train_loss)
	print(valid_loss)
	
	train_losses.append(train_loss)	
	train_global_losses.append(train_global_loss)
	train_local_losses.append(train_local_loss)
	train_fusion_losses.append(train_fusion_loss)
	train_accuracies.append(train_accuracy)

	valid_losses.append(valid_loss)
	valid_global_losses.append(valid_global_loss)
	valid_local_losses.append(valid_local_loss)
	valid_fusion_losses.append(valid_fusion_loss)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_loss < best_val_losses
	if is_best:
		print('Found Best')
		print(valid_loss)
		best_val_losses = valid_loss
		torch.save(global_model, os.path.join(PATH_OUTPUT, 'global_agcnn.pth'))
		torch.save(local_model, os.path.join(PATH_OUTPUT, 'local_agcnn.pth'))
		torch.save(fusion_model, os.path.join(PATH_OUTPUT, 'fusion_agcnn.pth'))
		print('Saved Best Models')

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

epoch_range = [1,2,3,4,5]
train_stats = pd.DataFrame(list(zip(epoch_range, train_accuracies, valid_accuracies, train_losses, valid_losses, valid_global_losses, valid_local_losses, valid_fusion_losses)), 
			   columns =['Epoch', 'Train Accuracy', 'Valid Accuracy', 'Train Loss', 'Valid Loss', 'Valid Global Loss', 'Valid Local Loss', 'Valid Fusion Loss'])
print('TRAINING SUMMARY')
print(train_stats.head(len(epoch_range)))
print('')
train_stats.to_csv ('output/agcnn_train_stats.csv', index = None, header=True)

best_fusion_model = torch.load(os.path.join(PATH_OUTPUT, 'fusion_agcnn.pth'))
best_global_model = torch.load(os.path.join(PATH_OUTPUT, 'global_agcnn.pth'))
best_local_model = torch.load(os.path.join(PATH_OUTPUT, 'local_agcnn.pth'))

best_global_model.to(device)
best_local_model.to(device)
best_fusion_model.to(device)

#FUSION_MODEL
##we evaluate using only the fusion model
test_loss, test_global_loss, test_local_loss, test_fusion_loss, test_accuracy, test_results = evaluate_agcnn(best_fusion_model, best_global_model, best_local_model, device, valid_loader, criterion)
print('Fusion Model')
print('Test Loss: ' + str(test_loss))
print('Test Accuracy: ' + str(test_accuracy))
print(' ')

test_prob, test_labels = predict_disease_agcnn(best_global_model, best_local_model, best_fusion_model, device, valid_loader, 'fusion')
np.save('output/validation_predictions_fusion_agcnn.npy', test_prob)
np.save('output/validation_labels_fusion_agcnn.npy', test_labels)

roc_auc_scores_list = []
for i in range(0,14):
	if test_labels[:,i].sum(axis=0) > 0 and test_labels[:,i].sum(axis=0) <= 234:
		roc_auc_scores_list.append(roc_auc_score(test_labels[:,i], test_prob[:,i]))
	else:
		roc_auc_scores_list.append(0.0)

class_names = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema'
				,'Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural_Effusion','Pleural_Other'
				,'Fracture','Support_Devices']

auc_df = pd.DataFrame(list(zip(class_names, roc_auc_scores_list)), 
			   columns =['ClassName', 'AUC'])
print(auc_df.head(14)) 
auc_df.to_csv ('output/auc_all_classes_fusion_agcnn.csv', index = None, header=True)

#GLOBAL
##we evaluate using only the fusion model
test_loss, test_global_loss, test_local_loss, test_fusion_loss, test_accuracy, test_results = evaluate_agcnn(best_fusion_model, best_global_model, best_local_model, device, valid_loader, criterion)
print('Global Model')
print('Test Loss: ' + str(test_loss))
print('Test Accuracy: ' + str(test_accuracy))
print(' ')

test_prob, test_labels = predict_disease_agcnn(best_global_model, best_local_model, best_fusion_model, device, valid_loader, 'global')
np.save('output/validation_predictions_global_agcnn.npy', test_prob)
np.save('output/validation_labels_global_agcnn.npy', test_labels)

roc_auc_scores_list = []
for i in range(0,14):
	if test_labels[:,i].sum(axis=0) > 0 and test_labels[:,i].sum(axis=0) <= 234:
		roc_auc_scores_list.append(roc_auc_score(test_labels[:,i], test_prob[:,i]))
	else:
		roc_auc_scores_list.append(0.0)

class_names = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema'
				,'Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural_Effusion','Pleural_Other'
				,'Fracture','Support_Devices']

auc_df = pd.DataFrame(list(zip(class_names, roc_auc_scores_list)), 
			   columns =['ClassName', 'AUC'])
print(auc_df.head(14)) 
auc_df.to_csv ('output/auc_all_classes_global_agcnn.csv', index = None, header=True)


#LOCAL
##we evaluate using only the fusion model
test_loss, test_global_loss, test_local_loss, test_fusion_loss, test_accuracy, test_results = evaluate_agcnn(best_fusion_model, best_global_model, best_local_model, device, valid_loader, criterion)
print('Local Model')
print('Test Loss: ' + str(test_loss))
print('Test Accuracy: ' + str(test_accuracy))
print(' ')

test_prob, test_labels = predict_disease_agcnn(best_global_model, best_local_model, best_fusion_model, device, valid_loader, 'local')
np.save('output/validation_predictions_local_agcnn.npy', test_prob)
np.save('output/validation_labels_local_agcnn.npy', test_labels)

roc_auc_scores_list = []
for i in range(0,14):
	if test_labels[:,i].sum(axis=0) > 0 and test_labels[:,i].sum(axis=0) <= 234:
		roc_auc_scores_list.append(roc_auc_score(test_labels[:,i], test_prob[:,i]))
	else:
		roc_auc_scores_list.append(0.0)

class_names = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema'
				,'Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural_Effusion','Pleural_Other'
				,'Fracture','Support_Devices']

auc_df = pd.DataFrame(list(zip(class_names, roc_auc_scores_list)), 
			   columns =['ClassName', 'AUC'])
print(auc_df.head(14)) 
auc_df.to_csv ('output/auc_all_classes_local_agcnn.csv', index = None, header=True)