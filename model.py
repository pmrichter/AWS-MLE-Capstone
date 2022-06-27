import torch.nn as nn

class LSTM(nn.Module):
	
	def __init__(self, hidden_size):
		super(LSTM, self).__init__()
		# there is only one time series
		input_size = 1
		# the output is only one predicted return
		output_size = 1
		# the model normally expects the batch size to be the second dimension, but we are putting it first so we have to let the model know.
		self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first = True)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, features, hidden = None):
		output, hidden = self.lstm(features, hidden)
		last_hidden_state = output[:, -1]
		prediction = self.fc(last_hidden_state)
		return prediction, hidden
