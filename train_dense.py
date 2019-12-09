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
from utils import train, evaluate, predict_disease, train_agcnn, predict_disease_multiclass
from plots import plot_learning_curves, plot_confusion_matrix
from mymodels import MyCNN, MyDenseNet, MyVGG16, MyResNet, MyAGCNN, MyDenseNetMultiClass, get_local_activated_patch, AGCNN_Fusion, MyCRAL
from torchsummary import summary

# Some parameters
MODEL_TYPE = 'DenseNet'
NUM_EPOCHS = 5
BATCH_SIZE = 32
USE_CUDA = False
NUM_WORKERS = 16
LR = 0.00001
MULTICLASS_FLAG = False
save_file = 'DenseNet_320.pth'
IMG_SIZE = 320

if MULTICLASS_FLAG == True:
	model = MyDenseNetMultiClass()
else:
	model = MyDenseNet()

# Path for saving model
PATH_OUTPUT = "output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

ROOT_DIR = "CheXpert-v1.0-small"
train_labels_file ='train.csv'
valid_labels_file ='valid.csv'

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

image_transformations = transforms.Compose([
   transforms.Resize((IMG_SIZE, IMG_SIZE)),
   #transforms.CenterCrop(224),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   normalize,
])

train_dataset = ImagesDataset(ROOT_DIR, train_labels_file, 'Ones', image_transformations, multiclass=MULTICLASS_FLAG)
valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=MULTICLASS_FLAG)
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
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion.to(device)
print('Model Summary')
print(summary(model, (3, IMG_SIZE, IMG_SIZE)))
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

best_val_losses = 1000000.00
train_losses, train_accuracies, train_aucs = [], [], []
valid_losses, valid_accuracies, valid_aucs = [], [], []
 
for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch, 10, False)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion, 10, False)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_loss < best_val_losses
	if is_best:
		print('Found Best')
		print(valid_loss)
		best_val_losses = valid_loss
		torch.save(model, os.path.join(PATH_OUTPUT, save_file))
		print('Saved Best Model')

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
best_model.to(device)

print('Train Accuracies')
print(train_accuracies)

print('Valid Accuracies')
print(valid_accuracies)

print('Train Losses')
print(train_losses)

print('Valid Losses')
print(valid_losses)

test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion,10,False)

print('Test Loss: ' + str(test_loss))
print('Test Accuracy: ' + str(test_accuracy))

if MULTICLASS_FLAG == False:
	test_prob, test_labels = predict_disease(best_model, device, valid_loader, MULTICLASS_FLAG)
else:
	test_prob, test_labels = predict_disease_multiclass(best_model, device, valid_loader)

np.save('output/validation_predictions.npy', test_prob)
np.save('output/validation_labels.npy', test_labels)

print('The shape of the predictions')
print(test_prob.shape)
print(test_labels.shape)
print(test_labels[0,:])
print(test_prob[0,:])

##AUC evaluation
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
auc_df.to_csv ('output/auc_all_classes_dense.csv', index = None, header=True)