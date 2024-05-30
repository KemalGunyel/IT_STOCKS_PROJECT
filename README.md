# IT_STOCKS_PROJECT
IT STOCK VISUALIZATION
# Interactive EDA for Tech Stocks

This project is an interactive Exploratory Data Analysis (EDA) application for major tech stocks using Streamlit. The application allows users to select different tech stocks, view various statistical plots, and analyze stock performance over time.

## Features

- Interactive selection of tech stocks and date ranges
- Plots for closing prices, trading volumes, moving averages (30-day and 90-day), and daily changes
- Optional plots for volatility, volume vs. closing price, RSI, Bollinger Bands, cumulative returns, and drawdowns
- Correlation matrix of daily returns
- Descriptive statistics for selected stocks
- Annotation of significant turning points

## Tech Stocks Included

- Apple (AAPL)
- Microsoft (MSFT)
- Google (GOOGL)
- Amazon (AMZN)
- Meta (META)
- Tesla (TSLA)
- Netflix (NFLX)
- NVIDIA (NVDA)
- AMD (AMD)
- Samsung (SSNLF)
- Cisco (CSCO)
- Oracle (ORCL)
- SAP (SAP)
- IBM (IBM)
- Intel (INTC)
- Salesforce (CRM)
- Adobe (ADBE)

## Usage
### Sidebar Controls
- Stock Selection: Select up to three tech stocks to analyze.
- Date Range: Choose a date range for the analysis.
- Turning Points: Select significant turning points to annotate on the plots.
- Optional Plots: Checkboxes to display additional plots (volatility, volume vs. closing price, RSI, Bollinger Bands, cumulative returns, and drawdowns).

## Main Plots
- Closing Prices: Line plot of the selected stocks' closing prices over time.
- Trading Volumes: Bar plot of the selected stocks' trading volumes over time.
- Correlation Matrix: Heatmap showing the correlation matrix of daily returns for all tech stocks.
- 30-Day and 90-Day Moving Averages: Line plots showing the moving averages.
- Daily Change Analysis: Box plot showing the daily change in price (close - open) by stock.

##Data Source
The stock data is fetched from Yahoo Finance using the yfinance Python library.

## Requirements
- Python 3.7+
- Streamlit
- Pandas
- Plotly
- yfinance

  
