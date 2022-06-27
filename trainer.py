import random
import torch
import copy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from loader import create_dataloaders
from model import LSTM
import matplotlib.pyplot as plt

def train_model(model, train_loader, validation_loader, test_loader, scaler, learning_rate, number_of_epochs, batch_size, verbose):

	best_model = None
	min_val_loss = None
	
	training_loss = []
	validation_loss = []
	
	best_model = None
	
	# Use Adam as optimizer
	optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
	# Use mean of squared errors
	criterion = torch.nn.MSELoss()
	
	for t in range(number_of_epochs):
	
			# set model to train mode
			model.train()
			# perform training for all the batches
			epoch_training_loss = 0
			for batch, (features, targets) in enumerate(train_loader, 1):
				# only use complete batches, so stop as soon as all complete batches have been processed
				if(batch > len(train_loader.dataset)//batch_size):
					break
				
				hidden = None
				prediction, hidden = model(features, hidden)
				optimizer.zero_grad()
				loss = criterion(prediction, targets)	
				epoch_training_loss+=loss
				loss.backward()
				optimizer.step()
	
			training_loss.append(epoch_training_loss.item())
		  
		  # after training, perform validation
			model.eval()
			epoch_validation_loss = 0
			with torch.no_grad():
				for batch, (features, targets) in enumerate(validation_loader, 1):
					# again, only use complete batches, so stop as soon as all complete batches have been processed
					if(batch > len(validation_loader.dataset)//batch_size):
						break
					
					hidden = None
					prediction, hidden = model(features, hidden)
					optimizer.zero_grad()
					loss = criterion(prediction, targets)	
					epoch_validation_loss+=loss
		   
			validation_loss.append(epoch_validation_loss.item())
	
			if min_val_loss == None or epoch_validation_loss.item() < min_val_loss:
				best_model = copy.deepcopy(model)
				min_val_loss = epoch_validation_loss.item()
	
			show_every_n_epochs = 10
			if verbose and t % show_every_n_epochs == 0:
				print(f'epoch {t}: train - {round(epoch_training_loss.item(), 4)}, val: - {round(epoch_validation_loss.item(), 4)}')
	
	best_model.eval()
	test_predicted = []
	test_actual = []
	with torch.no_grad():
		for batch, (features, targets) in enumerate(test_loader, 1):
			# again, only use complete batches, so stop as soon as all complete batches have been processed
			if(batch > len(test_loader.dataset)//batch_size):
				break
			
			hidden = None
			predictions, hidden = model(features, hidden)
			for prediction in predictions.tolist():
				unscaled_prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))[0][0]
				test_predicted.append(unscaled_prediction)
			for actual_value in targets.tolist():
				unscaled_actual = scaler.inverse_transform(np.array(actual_value).reshape(-1, 1))[0][0]
				test_actual.append(unscaled_actual)
	
	return best_model, training_loss, validation_loss, test_actual, test_predicted

def main():
	# just some testing code...
	
	# we want all results to be reproducible, so we set the random generator seeds
	random.seed(1)
	torch.manual_seed(1)
	
	train_loader, validation_loader, test_loader, scaler = create_dataloaders("AEP", 20, 50, 5000, 100)
	model = LSTM(hidden_size = 30)
	best_model, training_loss, validation_loss, test_actual, test_predicted = train_model(model, train_loader, validation_loader, test_loader, scaler, 0.01, 50, 100)
	
	plt.title('Test dataset')
	plt.plot(test_actual, label = 'actual')
	plt.plot(test_predicted, label = 'predicted')
	plt.legend()
	plt.show()
	
	plt.title('Training')
	plt.yscale('log')
	plt.plot(training_loss, label = 'training')
	plt.plot(validation_loss, label = 'validation')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend()
	plt.show()

if __name__ == "__main__":
    main()

