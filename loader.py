import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from os.path import exists

def load_series(ticker, days_return, time_series_length):
	# load dataframe from CSV file, convert it into shape (<time_series_length>,1) and take the last <time_series_length> data points.
	if check_existence(ticker, days_return) == True:
		filename=get_file_name(ticker, days_return)
		time_series = pd.read_csv(filename)['return'].astype(float).values.reshape(-1, 1)[-time_series_length:]
		return time_series
	else:
		return None

def check_existence(ticker, days_return):
	return exists(get_file_name(ticker, days_return))

def get_file_name(ticker, days_return):
	return f'./returns/{ticker}-{days_return}d-returns.csv'
	
# load training, validation and testing data (80% of the data goes into training, 10% into validation and 10% into testing)
def create_dataloaders(ticker, days_return, sequence_length, time_series_length, batch_size):
	
	time_series = load_series(ticker, days_return, time_series_length)
	# scale the series
	scaler = MinMaxScaler()
	time_series = scaler.fit_transform(time_series)		
	# arrange the data into feature values and target values by moving a window of size <sequence_length> through the data (starting at <sequence_length>, so each feature set is complete)
	features = []
	targets = []

	for i in range(sequence_length + 1, time_series_length + 1):
		features.append(time_series[i - (sequence_length + 1): i - 1])
		# the target is the next value right after the sequence.
		targets.append([time_series[i - 1]])

	number_of_sequences = len(features)

	# divide the total length into training, validation and testing data
	validation_start_index = round(number_of_sequences*0.8)
	test_start_index = validation_start_index + round(number_of_sequences*0.1)
	training_data_length = validation_start_index
	validation_data_length = test_start_index - validation_start_index
	test_data_length = number_of_sequences - test_start_index
	#print(f'Training data points: {training_data_length}')
	#print(f'Validation data points: {validation_data_length}')
	#print(f'Testing data points: {test_data_length}')

	# check the number of sequences
	check_number_of_sequences = training_data_length + validation_data_length + test_data_length
	if check_number_of_sequences != number_of_sequences:
		raise Exception(f'Error: total number of sequences is {check_number_of_sequences}, but should be {number_of_sequences}')
	
	train_loader = create_dataloader(features, targets, 0, validation_start_index, batch_size)
	validation_loader = create_dataloader(features, targets, validation_start_index, test_start_index, batch_size)
	test_loader = create_dataloader(features, targets, test_start_index, number_of_sequences, batch_size)

	return train_loader, validation_loader, test_loader, scaler

# create a dataloader with the given batch size
def create_dataloader(features, targets, start_index, end_index, batch_size):
	features_tensor = torch.tensor(data = np.array(features[start_index:end_index])).float()
	targets_tensor = torch.tensor(data = np.array(targets[start_index:end_index])).float()
	# adapt the shape
	targets_tensor = targets_tensor.squeeze(1)
	data_set = TensorDataset(features_tensor, targets_tensor)
	dataloader = torch.utils.data.DataLoader(data_set, batch_size = batch_size)

	return dataloader

def main():
	# do some testing
	train_loader, validation_loader, test_loader, scaler = create_dataloaders("AEP", 1, 50, 5000, 100)
			
	print(len(train_loader.dataset))
	print(len(validation_loader.dataset))
	print(len(test_loader.dataset))
	
	it = iter(train_loader)
	f, t = it.next()
	print(f.shape)
	print(t.shape)	

if __name__ == "__main__":
    main()
