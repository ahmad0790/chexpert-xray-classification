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
from utils import train, evaluate, predict_disease, train_agcnn
from plots import plot_learning_curves, plot_confusion_matrix
from mymodels import MyDenseNet, MyAGCNN, MyDenseNetMultiClass, get_local_activated_patch, AGCNN_Fusion, MyCRAL
from torchsummary import summary

# Some parameters
MODEL_TYPE = 'CRAL'
NUM_EPOCHS = 5
BATCH_SIZE = 32
USE_CUDA = True 
NUM_WORKERS = 16
LR = 0.00001
MULTICLASS_FLAG = False
save_file = 'cral_320.pth'
IMG_SIZE = 320

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
   transforms.Resize((IMG_SIZE,IMG_SIZE)),
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
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
model = MyCRAL()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion.to(device)
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

	print('Epoch Results')
	print(valid_loss)
	print(valid_accuracy)

	is_best = valid_loss < best_val_losses
	if is_best:
		print('Found Best')
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

##we evaluate using only the fusion model
test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion, 10, False)

print('Test Loss: ' + str(test_loss))
print('Test Accuracy: ' + str(test_accuracy))

test_prob, test_labels = predict_disease(best_model, device, valid_loader, MULTICLASS_FLAG)
np.save('output/validation_predictions.npy', test_prob)
np.save('output/validation_labels.npy', test_labels)

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
auc_df.to_csv ('output/auc_all_classes_cral.csv', index = None, header=True)

#train stats
epoch_range = [1,2,3,4,5]
train_stats = pd.DataFrame(list(zip(epoch_range, train_accuracies, valid_accuracies, train_losses, valid_losses)), 
			   columns =['Epoch', 'Train Accuracy', 'Valid Accuracy', 'Train Loss', 'Valid Loss'])
print('TRAINING SUMMARY')
print(train_stats.head(len(epoch_range)))
print('')
train_stats.to_csv ('output/cral_train_stats.csv', index = None, header=True)