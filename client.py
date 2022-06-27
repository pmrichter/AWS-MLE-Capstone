import random
import torch
import numpy as np
from trainer import train_model
from loader import create_dataloaders, load_series
from model import LSTM
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# we want all results to be reproducible, so we set the random generator seeds
random.seed(1)
torch.manual_seed(1)

def load_data(ticker, days_return, sequence_length, time_series_length, batch_size):
	# get the dataloaders and the used scaler (to be able to unscle the predictions afterwards)
	train_loader, validation_loader, test_loader, scaler = create_dataloaders(ticker, days_return, sequence_length, time_series_length, batch_size)
	return train_loader, validation_loader, test_loader, scaler

def create_model(hidden_size):
	# set up the model
	model = LSTM(hidden_size = hidden_size)
	return model	

def train(model, train_loader, validation_loader, test_loader, scaler, learning_rate, number_of_epochs, batch_size, verbose):
	best_model, training_loss, validation_loss, test_actual, test_predicted = train_model(model, train_loader, validation_loader, test_loader, scaler, learning_rate, number_of_epochs, batch_size, verbose)
	return 	best_model, training_loss, validation_loss, test_actual, test_predicted
	
def plot_training_progress(test_actual, test_predicted, training_loss, validation_loss, ticker, days_return):
	
	plt.title(f'Test dataset [ticker={ticker}, {days_return}d return]')
	plt.plot(test_actual, label = 'actual')
	plt.plot(test_predicted, label = 'predicted')
	plt.legend()
	plt.show()
	
	plt.title(f'Training [ticker={ticker}, {days_return}d return]')
	plt.yscale('log')
	plt.plot(training_loss, label = 'training')
	plt.plot(validation_loss, label = 'validation')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend()
	plt.show()
	
def process_ticker(ticker, days_return, sequence_length, time_series_length, hidden_size, batch_size, learning_rate, number_of_epochs, plot, verbose):
	
	if verbose == True:
		print(f"Processing {days_return}d return for ticker {ticker}.")
	
	train_loader, validation_loader, test_loader, scaler = load_data(ticker, days_return, sequence_length, time_series_length, batch_size)
	model = create_model(hidden_size)
	best_model, training_loss, validation_loss, test_actual, test_predicted = train(model, train_loader, validation_loader, test_loader, scaler, learning_rate, number_of_epochs, batch_size, verbose)
	if plot == True:
		plot_training_progress(test_actual, test_predicted, training_loss, validation_loss, ticker, days_return)
	
	return test_actual, test_predicted

# create a recommendation based on the predicted return
def create_recommendation(predicted_returns, neutral_range):
	recommendations = []
	for predicted_return in predicted_returns:
		if predicted_return>=neutral_range:
			recommendations.append('buy')
		elif predicted_return<=-neutral_range:
			recommendations.append('sell')
		else:
			recommendations.append('neutral')
	
	return recommendations


# based on the recommendations and the actual future returns, evaluate the recommendations
def evaluate_recommendations(recommendations, actual_returns, perfect_recommendations, verbose):
	
	n_of_predictions = len(recommendations)

	n_buy=0
	n_sell=0
	n_neutral=0
	n_buy_correct=0
	n_sell_correct=0
	n_neutral_correct=0
	n_buy_incorrect_neutral=0
	n_buy_incorrect_sell=0
	n_sell_incorrect_neutral=0
	n_sell_incorrect_buy=0
	n_neutral_incorrect_buy=0
	n_neutral_incorrect_sell=0

	profit_loss=1
	
	for i in range(len(recommendations)):
	
		pred = recommendations[i]
		act=perfect_recommendations[i]
		actual_return=actual_returns[i]
		if pred == 'buy':
			n_buy+=1
			profit_loss*=(1+actual_return)
			if act == 'buy':
				n_buy_correct+=1
			elif act == 'neutral':
				n_buy_incorrect_neutral+=1
			elif act == 'sell':
				n_buy_incorrect_sell+=1
		elif pred == 'neutral':
			n_neutral+=1
			if act == 'buy':
				n_neutral_incorrect_buy+=1
			elif act == 'neutral':
				n_neutral_correct+=1
			elif act == 'sell':
				n_neutral_incorrect_sell+=1
		elif pred == 'sell':
			n_sell+=1
			profit_loss*=(1-actual_return)
			if act == 'buy':
				n_sell_incorrect_buy+=1
			elif act == 'neutral':
				n_sell_incorrect_neutral+=1
			elif act == 'sell':
				n_sell_correct+=1

	n_correct = n_buy_correct + n_neutral_correct + n_sell_correct
	accuracy = n_correct / n_of_predictions

	if verbose == True:
		print(f"Number of predictions: {n_of_predictions}")
		print(f"Number of correct predictions: {n_correct}")
		print(f"Recommendation accuracy: {accuracy}")
		
		print(f"Number of BUY recommendations: {n_buy}")
		print(f"Number of NEUTRAL recommendations: {n_neutral}")
		print(f"Number of SELL recommendations: {n_sell}")
		
		print(f"Number of correct BUY recommendations: {n_buy_correct}")
		print(f"Number of correct SELL recommendations: {n_sell_correct}")
		print(f"Number of correct NEUTRAL recommendations: {n_neutral_correct}")
		
		print(f"Number of incorrect BUY recommendations (where NEUTRAL would be correct): {n_buy_incorrect_neutral}")
		print(f"Number of incorrect BUY recommendations (where SELL would be correct): {n_buy_incorrect_sell}")
		
		print(f"Number of incorrect SELL recommendations (where NEUTRAL would be correct): {n_sell_incorrect_neutral}")
		print(f"Number of incorrect SELL recommendations (where BUY would be correct): {n_sell_incorrect_buy}")
		
		print(f"Number of incorrect NEUTRAL recommendations (where BUY would be correct): {n_neutral_incorrect_buy}")
		print(f"Number of incorrect NEUTRAL recommendations (where SELL would be correct): {n_neutral_incorrect_sell}")
	
	return profit_loss - 1, accuracy

# calculate the simple moving average over a given number of days
def calculate_simple_moving_average(values, length):

	sma=[]
	
	for i in range(len(values) - length):
		sma.append(sum(values[i:i+length]) / length)
	
	return sma

def load_tickers():
	# get all tickers from file
	# use this file with only one ticker for debugging
	# filename='SP500-tickers-test.csv'
	# full list
	filename='SP500-tickers.csv'
	df=pd.read_csv(filename)
	ticker_list=df['TICKER'].values.tolist()
	#print(ticker_list)
	ticker_list_filtered = []
	for ticker in ticker_list:
		series = load_series(ticker, 1, 5000)
		if series is not None and len(series)>=5000:
			ticker_list_filtered.append(ticker)

	return ticker_list_filtered
	
def process_single(ticker, sequence_length, time_series_length, hidden_size, batch_size, learning_rate, number_of_epochs, neutral_range, verbose):

	test_actual_1d, test_predicted_1d = process_ticker(ticker, 1, sequence_length, time_series_length, hidden_size, batch_size, learning_rate, number_of_epochs, verbose, verbose)
	_, test_predicted_5d = process_ticker(ticker, 5, sequence_length, time_series_length, hidden_size, batch_size, learning_rate, number_of_epochs, verbose, verbose)
	_, test_predicted_20d = process_ticker(ticker, 20, sequence_length, time_series_length, hidden_size, batch_size, learning_rate, number_of_epochs, verbose, verbose)
	
	test_predicted_1d = np.array(test_predicted_1d)
	test_predicted_5d = np.array(test_predicted_5d)
	test_predicted_20d = np.array(test_predicted_20d)
	
	lstm_prediction = (test_predicted_1d + ((test_predicted_5d + 1)**(1/5) - 1) + ((test_predicted_20d + 1)**(1/20) - 1))/3
	
	perfect_recommendations = create_recommendation(test_actual_1d, neutral_range)
	sma18_prediction = calculate_simple_moving_average(test_actual_1d, 18)[-len(test_predicted_1d):]
	sma18_recommendations = create_recommendation(sma18_prediction, neutral_range)
	lstm_recommendations = create_recommendation(lstm_prediction, neutral_range)
	
	profit_loss_lstm, accuracy_lstm = evaluate_recommendations(lstm_recommendations, test_actual_1d, perfect_recommendations, verbose)
	profit_loss_sma18, accuracy_sma18 = evaluate_recommendations(sma18_recommendations, test_actual_1d, perfect_recommendations, verbose)

	if verbose==True:
		print('\nEVALUATION OF LSTM PREDICTIONS:')
		print(f'profit/loss={profit_loss_lstm*100}%')
		
		print('\nEVALUATION OF 18 DAY SIMPLE MOVING AVERAGE PREDICTIONS (BENCHMARK):')
		print(f'profit/loss={profit_loss_sma18*100}%')
		
		print('\nCONCLUSION:')
		if profit_loss_lstm>profit_loss_sma18:
			print(f'LSTM beats SMA18 for ticker {ticker}.')
		elif profit_loss_lstm>profit_loss_sma18:
			print(f'LSTM and SMA18 perform equally well for ticker {ticker}.')
		else:
			print(f'SMA18 beats LSTM for ticker {ticker}.')
	
	return profit_loss_lstm, profit_loss_sma18, accuracy_lstm, accuracy_sma18

def main():

	# Get cmd line parameters
	parser = argparse.ArgumentParser(description='LSTM training and prediction for stock return series in the S&P500 index.')

	parser.add_argument('--ticker', action='store', dest='ticker', help='Ticker from S&P500 to be used.')
	parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.01, help='Learning rate')
	parser.add_argument('--neutral_range', action='store', dest='neutral_range', type=float, default=0.0005, help='A return below neutral_range is considered neutral, i.e. neither a BUY nor a SELL recommendation')
	parser.add_argument('--sequence_length', action='store', dest='sequence_length', type=int, default=50, help='Length of the input sequences for the LSTM')
	parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=100, help='Batch size for training')
	parser.add_argument('--hidden_dim', action='store', dest='hidden_dim', type=int, default=30, help='Hidden dimension of LSTM')
	parser.add_argument('--training_epochs', action='store', dest='training_epochs', type=int, default=100, help='Number of training epochs')
	
	in_arg=parser.parse_args()
	
	print("===Input parameters===")
	print(f"Ticker: {in_arg.ticker}")
	print(f"Learning rate: {in_arg.learning_rate}")
	print(f"Neutral range: {in_arg.neutral_range}")
	print(f"Sequence size: {in_arg.sequence_length}")
	print(f"Batch size: {in_arg.batch_size}")
	print(f"LSTM hidden dimensions: {in_arg.hidden_dim}")
	print(f"Training epochs: {in_arg.training_epochs}")
	print("======================")

	# LSTM input sequence size
	sequence_length = in_arg.sequence_length
	# LSTM hidden layer size
	hidden_size = in_arg.hidden_dim
	# learning rate
	learning_rate = in_arg.learning_rate
	# number of training epochs
	number_of_epochs = in_arg.training_epochs
	# ticker
	ticker = in_arg.ticker
	# total time series length
	time_series_length=5000
	# batch size
	batch_size = in_arg.batch_size
	# neutral range
	neutral_range=in_arg.neutral_range

	all_tickers = load_tickers()
	
	if ticker != None:	
		if ticker not in all_tickers:
			raise Exception(f'Error: Ticker {ticker} is not part of the S&P 500 or does not have complete data for the last 20 years.')
		
		process_single(ticker, sequence_length, time_series_length, hidden_size, batch_size, learning_rate, number_of_epochs, neutral_range, True)
	
	else:
		# perform prediction for all 20 random available tickers from the S&P500 index
		print("Randomly picking 50 tickers...")
		selected_tickers = random.choices(all_tickers, k=50)
		print("ticker, accuracy_lstm, accuracy_sma18, profit_loss_lstm, profit_loss_sma18")
		for ticker in selected_tickers:
			profit_loss_lstm, profit_loss_sma18, accuracy_lstm, accuracy_sma18 = process_single(ticker, sequence_length, time_series_length, hidden_size, batch_size, learning_rate, number_of_epochs, neutral_range, False)
			print(f"{ticker}, {accuracy_lstm}, {accuracy_sma18}, {profit_loss_lstm}, {profit_loss_sma18}")
	
if __name__ == "__main__":
    main()
