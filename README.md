This project is about building a tool to predict stock market returns by using LSTM networks.

Results: The model predictions are not better than 18 day simple moving averages, so LSTM predictions based on past returns is not enough to make a profit.

used libraries:
- pytorch 1.11
- scikit-learn 1.1.1
- matplotlib 3.5.2
- argparse 1.4.0
- pandas 1.4.3
- yahoofinancials 1.6
- numpy 1.23.0

Files:
- AEP-visualization.xlsx: visualization of stock prices for ticker AEP
- client.py: main client application
- first-validation.ipynb: Jupyter notebook for first validation of downloaded stock price data
- first-validation.html: HTML version of the above
- INTC-training-evaluation.txt: Command-line output for training of INTC ticker
- WMT-training-evaluation.txt: Command-line output for training of WMT ticker
- INTC-visualization.xlsx: visualization of 1d, 5d and 20d returns for ticker INTC
- loader.py: functions for dataloading
- model.py: mode class
- preprocess.py: functions for data preprocessing
- price-loader.py: functions for downloading price data from Yahoo Finance
- proposal.pdf: Original project proposal
- random-50.csv: Prediction results for a random selection of 50 tickers
- screenshots folder: several screenshots documenting important steps
- SP500-tickers.csv: CSV file with all S&P500 tickers downloaded from Wikipedia
- SP500-tickers.xlsx: Excel version of the above
- trainer.py: functions for model training
- validation.ipynb: Jupyter notebook for validation of preprocessing results
- validation.html: HTML version of the above
