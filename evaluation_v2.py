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

USE_CUDA = True 
MODEL_TYPE = 'CRAL_ATT1_MULTI'
BATCH_SIZE = 32
NUM_WORKERS = 4

ROOT_DIR = "CheXpert-v1.0-small"
valid_labels_file ='valid.csv'

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

image_transformations = transforms.Compose([
   transforms.Resize((320,320)),
   transforms.ToTensor(),
   normalize
])

PATH_OUTPUT = "output/"

if MODEL_TYPE == 'CRAL_ATT1_MULTI':
	save_file = 'cral_att1_320_multi.pth'

elif MODEL_TYPE == 'CRAL_ATT1_ONES':
	save_file = 'cral_att1_320_ones.pth'

elif MODEL_TYPE == 'CRAL_ATT2_MULTI':
	save_file = 'cral_att2_320_multi.pth'

elif MODEL_TYPE == 'DENSE_ONES':
	save_file = 'densenet_320_ones.pth'

elif MODEL_TYPE == 'DENSE_MULTI':
	save_file = 'densenet_320_multi.pth'

elif MODEL_TYPE == 'AGCNN_ONES':
	fusion_save_file = 'fusion_320_agcnn_ones.pth'
	global_save_file = 'global_320_agcnn_ones.pth'
	local_save_file = 'local_320_agcnn_ones.pth'

else:
	raise AssertionError("Wrong Model Type!")

print('Evaluating ' + str(MODEL_TYPE))

if MODEL_TYPE in ('CRAL_ATT1_MULTI','CRAL_ATT2_MULTI','DENSE_MULTI'):
	
	MULTICLASS_FLAG = True

	device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
	torch.manual_seed(1)
	if device == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=True)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	criterion = nn.BCELoss()
	criterion.to(device)
	best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
	best_model.to(device)

	test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion, 10)
	test_prob, test_labels = predict_disease(best_model, device, valid_loader, True)

	pred_file = 'output/validation_predictions_' + str(MODEL_TYPE) + '.npy'
	np.save(pred_file, test_prob)
	np.save('output/validation_labels.npy', test_labels)

elif MODEL_TYPE in ('CRAL_ATT1_ONES','DENSE_ONES'):

	device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
	torch.manual_seed(1)
	if device == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=False)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)
	best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
	best_model.to(device)

	test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion, 10)
	test_prob, test_labels = predict_disease(best_model, device, valid_loader, False)

	pred_file = 'output/validation_predictions_' + str(MODEL_TYPE) + '.npy'
	np.save(pred_file, test_prob)
	np.save('output/validation_labels.npy', test_labels)

elif MODEL_TYPE in ('AGCNN_ONES'):
	
	MULTICLASS_FLAG = False
	valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=True)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
	torch.manual_seed(1)
	if device == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)
	best_global_model = torch.load(os.path.join(PATH_OUTPUT, global_save_file))
	best_fusion_model = torch.load(os.path.join(PATH_OUTPUT, fusion_save_file))
	best_local_model = torch.load(os.path.join(PATH_OUTPUT, local_save_file))
	best_global_model.to(device)
	best_fusion_model.to(device)
	best_local_model.to(device)

	test_loss, test_accuracy, test_results = evaluate_agcnn(best_model, device, valid_loader, criterion, 10)

	test_prob, test_labels = predict_disease_agcnn(best_model, device, valid_loader, True)

	pred_file = 'output/validation_predictions_' + str(MODEL_TYPE) + '.npy'
	np.save(pred_file, test_prob)
	np.save('output/validation_labels.npy', test_labels)

###################################### REPORTING PERFORMANCE ################################################

print('Test Loss: ' + str(test_loss))
print('Test Accuracy: ' + str(test_accuracy))

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
auc_df.to_csv ('output/auc_all_classes.csv', index = None, header=True)