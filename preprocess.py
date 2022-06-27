import pandas as pd
from os.path import exists

#This tool calculates daily, weekly and monthly returns. Because on weekends and holidays the stock exchanges are closed anyway, these do not count and we can simplify the returns calculation to weekly=5 days and monthly=20 days.

def load_series(ticker):
	path=f"./prices/{ticker}.csv"
	if exists(path):
		original=pd.read_csv(path)
		#set the column headers
		original.columns=['datetime','price']
		original['datetime']=pd.to_datetime(original['datetime'])
		
		df=original['price']
		df.index=original['datetime']
	
		return df
	else:
		return None

def calculate_returns(df, days, points):
	#calculate the percentage change between points with given distance
	returns=df.pct_change(periods=days)
	returns.rename( "return", inplace=True)
	returns=returns[-points:]
	#get only the last data points to make sure all dataframes have the same length
	return returns
	
def save_series(ticker, df1, df5, df20):
	path1=f"./returns/{ticker}-1d-returns.csv"
	df1.to_csv(path1, header=True, index=True)
	path5=f"./returns/{ticker}-5d-returns.csv"
	df5.to_csv(path5, header=True, index=True)
	path20=f"./returns/{ticker}-20d-returns.csv"
	df20.to_csv(path20, header=True, index=True)

def load_tickers():
	#get all tickers from file
	filename='SP500-tickers.csv'
	df=pd.read_csv(filename)
	ticker_list=df['TICKER'].values.tolist()
	return ticker_list

points=5001
tickers=load_tickers()
for ticker in tickers:
	df=load_series(ticker)
	if df is not None:
		df1=calculate_returns(df,1, points)
		df5=calculate_returns(df,5, points)
		df20=calculate_returns(df,20, points)

		save_series(ticker, df1, df5, df20)
