from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np

def load_historical_data(tickers):
	for ticker in tickers:
		#print(f"Loading historical price data for ticker {ticker}")
		yahoo_financials = YahooFinancials(ticker)
		#get the 20 calendar years from 2002-2021
		prices = yahoo_financials.get_historical_price_data('2002-06-11', '2022-06-10', 'daily')
		prices_df=convert_prices(prices,ticker)
		print(f"Loaded {len(prices_df)} historical prices for ticker {ticker}")
		#ignore series shorter than 5036, which is the length of a complete series.
		if(len(prices_df) >= 5036):
			save_prices(prices_df,ticker,"./output")
		else:
			print(f"Ticker {ticker} only has {len(prices_df)} records, so it is ignored. ")

def convert_prices(price_records, ticker):
	#print('convert prices')
	dates=[]
	prices=[]
	
	if "prices" in price_records[ticker]:
		for p in price_records[ticker]["prices"]:
			dates.append(p['formatted_date'])
			prices.append(p['adjclose'])
	else:
		print(f"No price data available for ticker {ticker}, so it is ignored. ")
		
	raw={'date':dates, 'prices':prices}
	df = pd.DataFrame(raw)
	#print(df)
	return df
				
def save_prices(df, ticker, target_folder):
	#print('save file')
	df.to_csv(target_folder+"/"+ticker+".csv", header=False, index=False)

def load_tickers(test):
	#get all tickers from file
	filename='SP500-tickers.csv'
	if(test):
		filename='SP500-tickers-test.csv'
	df=pd.read_csv(filename)
	ticker_list=df['TICKER'].values.tolist()
	return ticker_list

ticker_list=load_tickers(False)
load_historical_data(ticker_list)

print(ticker_list)
