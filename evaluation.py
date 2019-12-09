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
from utils import train, evaluate, predict_disease, train_agcnn, evaluate_agcnn, predict_disease_agcnn
from plots import plot_learning_curves, plot_confusion_matrix
from mymodels import MyDenseNet, MyAGCNN, MyDenseNetMultiClass, get_local_activated_patch, AGCNN_Fusion, MyCRAL
import sys

#pass model_type as argument
USE_CUDA = True 
MODEL_TYPE = sys.argv[1]
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

PATH_OUTPUT = "model_output/"

if MODEL_TYPE == 'CRAL_ATT1_MULTI':
	save_file = 'cral_att1_320_multi.pth'
	MULTICLASS_FLAG = True
	from mycral_att1_multi import MyCRAL 

elif MODEL_TYPE == 'CRAL_ATT1_ONES':
	save_file = 'cral_att1_320_ones.pth'
	MULTICLASS_FLAG = False
	from mymodels import MyCRAL

elif MODEL_TYPE == 'CRAL_ATT2_MULTI':
	save_file = 'cral_att2_320_multi.pth'
	MULTICLASS_FLAG = True
	from mycral_att2_multi import MyCRAL 

elif MODEL_TYPE == 'CRAL_ATT2_ONES':
	save_file = 'cral_320_att2_ones.pth'
	MULTICLASS_FLAG = False
	from mycral_att2_ones import MyCRAL

elif MODEL_TYPE == 'DENSE_MULTI':
	save_file = 'densenet_320_multi.pth'
	MULTICLASS_FLAG = True

elif MODEL_TYPE == 'DENSE_ONES':
	save_file = 'densenet_320_ones.pth'
	MULTICLASS_FLAG = False
	from mymodels import MyDenseNet

elif MODEL_TYPE == 'AGCNN_MULTI':
	fusion_save_file = 'fusion_320_agcnn_multi.pth'
	global_save_file = 'global_320_agcnn_multi.pth'
	local_save_file = 'local_320_agcnn_multi.pth'
	MULTICLASS_FLAG = True

elif MODEL_TYPE == 'AGCNN_ONES':
	fusion_save_file = 'fusion_320_agcnn_ones.pth'
	global_save_file = 'global_320_agcnn_ones.pth'
	local_save_file = 'local_320_agcnn_ones.pth'
	MULTICLASS_FLAG = False
	BATCH_SIZE = 16

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

	valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=MULTICLASS_FLAG)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)
	best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
	best_model.to(device)

	test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion, 10, MULTICLASS_FLAG)
	test_prob, test_labels = predict_disease(best_model, device, valid_loader, MULTICLASS_FLAG)

	pred_file = 'model_output/validation_predictions_' + str(MODEL_TYPE) + '.npy'
	np.save(pred_file, test_prob)
	np.save('model_output/validation_labels.npy', test_labels)

elif MODEL_TYPE in ('CRAL_ATT1_ONES','DENSE_ONES','CRAL_ATT2_ONES'):

	device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
	torch.manual_seed(1)
	if device == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=MULTICLASS_FLAG)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)
	best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
	best_model.to(device)
	best_model.to(device)

	test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion, 10, MULTICLASS_FLAG)
	test_prob, test_labels = predict_disease(best_model, device, valid_loader, MULTICLASS_FLAG)

	pred_file = 'model_output/validation_predictions_' + str(MODEL_TYPE) + '.npy'
	np.save(pred_file, test_prob)
	np.save('model_output/validation_labels.npy', test_labels)

elif MODEL_TYPE in ('AGCNN_ONES'):
	
	valid_dataset = ImagesDataset(ROOT_DIR, valid_labels_file, 'Ones', image_transformations, multiclass=MULTICLASS_FLAG)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
	torch.manual_seed(1)
	if device == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	criterion = nn.BCELoss()
	criterion.to(device)
	best_global_model = torch.load(os.path.join(PATH_OUTPUT, global_save_file))
	best_fusion_model = torch.load(os.path.join(PATH_OUTPUT, fusion_save_file))
	best_local_model = torch.load(os.path.join(PATH_OUTPUT, local_save_file))
	best_global_model.to(device)
	best_fusion_model.to(device)
	best_local_model.to(device)

	test_loss, test_global_loss, test_local_loss, test_fusion_loss, test_accuracy, test_results = evaluate_agcnn(best_fusion_model, best_global_model, best_local_model, device, valid_loader, criterion)
	test_prob, test_labels = predict_disease_agcnn(best_global_model, best_local_model, best_fusion_model, device, valid_loader, 'fusion')

	pred_file = 'model_output/validation_predictions_' + str(MODEL_TYPE) + '.npy'
	np.save(pred_file, test_prob)
	np.save('model_output/validation_labels.npy', test_labels)

###################################### REPORTING PERFORMANCE ################################################
print(' ')
print('REPORTING MODEL: ' + str(MODEL_TYPE) + ' PERFORMANCE ON VALIDATION SET')
print('Validation Loss: ' + str(test_loss))
print('Validation Accuracy: ' + str(test_accuracy))

##AUC evaluation
roc_auc_scores_list = []
for i in range(0,14):
	if test_labels[:,i].sum(axis=0) > 0 and test_labels[:,i].sum(axis=0) <= 234:
		roc_auc_scores_list.append(roc_auc_score(test_labels[:,i], test_prob[:,i]))

class_names = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema'
				,'Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural_Effusion','Pleural_Other'
				,'Support_Devices']

auc_df = pd.DataFrame(list(zip(class_names, roc_auc_scores_list)), 
               columns =['ClassName', 'AUC'])
print('Average 14 Class AUC: ' + str(auc_df['AUC'].mean()))
print('Average 5 Class AUC: ' + str(auc_df.iloc[[2,5,6,8,10],1].mean()))
print(auc_df.head(14)) 
auc_df.to_csv ('model_output/auc_all_classes_' + str(MODEL_TYPE) + '.csv', index = None, header=True)