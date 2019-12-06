import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, model='_'):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation Loss')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig('output/loss_curve' + str(model) + '.png')
	#plt.show()

	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train Accuracy')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(loc="best")
	
	plt.savefig('output/accuracy_curve' + str(model) + '.png')
	#plt.show()


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	
	#code credit: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

	results = np.array(results)
	y_true = results[:,0]
	y_pred = results[:,1]
	cm = confusion_matrix(y_true, y_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	classes = class_names

	cmap=plt.cm.Blues
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]),
	       yticks=np.arange(cm.shape[0]),
	       xticklabels=classes, 
	       yticklabels=classes,
	       title='Confusion Matrix for Model',
	       ylabel='True label',
	       xlabel='Predicted label')

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
	    for j in range(cm.shape[1]):
	        ax.text(j, i, format(cm[i, j], fmt),
	                ha="center", va="center",
	                color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()

	plt.savefig('output/confusion_matrix.png')
	#fig.show()


