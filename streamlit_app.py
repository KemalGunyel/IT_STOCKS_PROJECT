import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from time import sleep

# List of tech stock symbols
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'AMD', 'SSNLF', 'CSCO', 'ORCL', 'SAP', 'IBM', 'INTC', 'CRM', 'ADBE']
START_DATE = '2018-01-01'
END_DATE = '2024-05-05'

# Define turning points
turning_points = {
    'COVID-19 Announcement': '2020-03-11',
    'ChatGPT Announcement': '2022-11-30',
    'Stock Market Crash': '2020-02-20',
    'Vaccination Rollout': '2020-12-14'
}

def fetch_stock_data(symbol, start_date, end_date, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            df['Symbol'] = symbol
            df.reset_index(inplace=True)  # Ensure 'Date' is a column, not the index
            return df
        except Exception as e:
            attempt += 1
            if attempt == retries:
                st.error(f"Error fetching data for {symbol}: {e}")
            else:
                sleep(1)  # Wait a bit before retrying
    return pd.DataFrame()

def load_data():
    combined_df = pd.DataFrame()
    for stock in tech_stocks:
        df = fetch_stock_data(stock, START_DATE, END_DATE)
        if not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

combined_df = load_data()

# Ensure 'Date' is in datetime format
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Calculate daily returns
combined_df['Daily Return'] = combined_df.groupby('Symbol')['Close'].pct_change()

# Calculate daily price change
combined_df['Daily Change'] = combined_df['Close'] - combined_df['Open']

# Calculate monthly average returns
combined_df['Month'] = combined_df['Date'].dt.to_period('M')
monthly_avg_returns = combined_df.groupby(['Month', 'Symbol'])['Daily Return'].mean().reset_index()
monthly_avg_returns['Month'] = monthly_avg_returns['Month'].astype(str)  # Convert Period to string

# Calculate annual volatility
combined_df['Year'] = combined_df['Date'].dt.year
annual_volatility = combined_df.groupby(['Year', 'Symbol'])['Daily Return'].std().reset_index()

# Correlation Matrix of Daily Returns
correlation_matrix = combined_df.pivot_table(values='Daily Return', index='Date', columns='Symbol').corr()

# Moving Averages: 30 days and 90 days
for symbol in combined_df['Symbol'].unique():
    combined_df.loc[combined_df['Symbol'] == symbol, 'MA30'] = combined_df[combined_df['Symbol'] == symbol]['Close'].rolling(window=30).mean()
    combined_df.loc[combined_df['Symbol'] == symbol, 'MA90'] = combined_df[combined_df['Symbol'] == symbol]['Close'].rolling(window=90).mean()

# Volatility Analysis: Calculate rolling standard deviation of daily returns
combined_df['Volatility'] = combined_df.groupby('Symbol')['Daily Return'].rolling(window=30).std().reset_index(0, drop=True)

# Calculate Relative Strength Index (RSI)
def calculate_rsi(data, window):
    diff = data.diff(1).dropna()
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

for symbol in combined_df['Symbol'].unique():
    combined_df.loc[combined_df['Symbol'] == symbol, 'RSI'] = calculate_rsi(combined_df[combined_df['Symbol'] == symbol]['Close'], window=14)

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band

for symbol in combined_df['Symbol'].unique():
    rolling_mean, upper_band, lower_band = calculate_bollinger_bands(combined_df[combined_df['Symbol'] == symbol]['Close'], window=20)
    combined_df.loc[combined_df['Symbol'] == symbol, 'Bollinger_Middle'] = rolling_mean
    combined_df.loc[combined_df['Symbol'] == symbol, 'Bollinger_Upper'] = upper_band
    combined_df.loc[combined_df['Symbol'] == symbol, 'Bollinger_Lower'] = lower_band

# Calculate Cumulative Returns
combined_df['Cumulative Return'] = combined_df.groupby('Symbol')['Daily Return'].cumsum()

# Calculate Drawdowns
def calculate_drawdown(data):
    cumulative = (1 + data).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown

for symbol in combined_df['Symbol'].unique():
    combined_df.loc[combined_df['Symbol'] == symbol, 'Drawdown'] = calculate_drawdown(combined_df[combined_df['Symbol'] == symbol]['Daily Return'])

# Streamlit app layout
st.title("Interactive EDA for Tech Stocks")

# Sidebar for symbol selection
st.sidebar.header("Stock Selection")
symbols = combined_df['Symbol'].unique()
symbol1 = st.sidebar.selectbox("Select First Symbol", symbols, index=0)
symbol2 = st.sidebar.selectbox("Select Second Symbol", symbols, index=1)
symbol3 = st.sidebar.selectbox("Select Third Symbol", symbols, index=2)

# Date range input using st.date_input in the sidebar
min_date = combined_df['Date'].min()
max_date = combined_df['Date'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter data based on selections
start_date, end_date = date_range
filtered_df1 = combined_df[(combined_df['Symbol'] == symbol1) & 
                           (combined_df['Date'] >= pd.to_datetime(start_date)) & 
                           (combined_df['Date'] <= pd.to_datetime(end_date))]

filtered_df2 = combined_df[(combined_df['Symbol'] == symbol2) & 
                           (combined_df['Date'] >= pd.to_datetime(start_date)) & 
                           (combined_df['Date'] <= pd.to_datetime(end_date))]

filtered_df3 = combined_df[(combined_df['Symbol'] == symbol3) & 
                           (combined_df['Date'] >= pd.to_datetime(start_date)) & 
                           (combined_df['Date'] <= pd.to_datetime(end_date))]

# Sidebar for selecting turning points to display
st.sidebar.header("Turning Points")
selected_turning_points = st.sidebar.multiselect(
    "Select Turning Points",
    options=list(turning_points.keys()),
    default=list(turning_points.keys())
)

# Create combined plot for closing prices
fig = go.Figure()

fig.add_trace(go.Scatter(x=filtered_df1['Date'], y=filtered_df1['Close'], mode='lines', name=f'{symbol1} Closing Prices'))
fig.add_trace(go.Scatter(x=filtered_df2['Date'], y=filtered_df2['Close'], mode='lines', name=f'{symbol2} Closing Prices'))
fig.add_trace(go.Scatter(x=filtered_df3['Date'], y=filtered_df3['Close'], mode='lines', name=f'{symbol3} Closing Prices'))

# Add turning points annotations
for event in selected_turning_points:
    date = turning_points[event]
    fig.add_vline(x=pd.to_datetime(date), line=dict(color='Red', dash='dash'))
    fig.add_annotation(x=pd.to_datetime(date), y=max(filtered_df1['Close'].max(), filtered_df2['Close'].max(), filtered_df3['Close'].max()),
                       text=event, showarrow=True, arrowhead=1)

fig.update_layout(title='Closing Prices Over Time', xaxis_title='Date', yaxis_title='Close Price')
st.plotly_chart(fig)

# Create combined plot for volumes
fig_volume = go.Figure()

fig_volume.add_trace(go.Bar(x=filtered_df1['Date'], y=filtered_df1['Volume'], name=f'{symbol1} Volume'))
fig_volume.add_trace(go.Bar(x=filtered_df2['Date'], y=filtered_df2['Volume'], name=f'{symbol2} Volume'))
fig_volume.add_trace(go.Bar(x=filtered_df3['Date'], y=filtered_df3['Volume'], name=f'{symbol3} Volume'))

fig_volume.update_layout(title='Trading Volume Over Time', xaxis_title='Date', yaxis_title='Volume', barmode='group')
st.plotly_chart(fig_volume)

# Correlation Matrix of Daily Returns
st.header("Correlation Matrix of Daily Returns")
correlation_matrix = combined_df.pivot_table(values='Daily Return', index='Date', columns='Symbol').corr()
fig_corr = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix of Daily Returns', labels=dict(color="Correlation"))
st.plotly_chart(fig_corr)

# Create combined plot for 30-day moving averages
fig_ma30 = go.Figure()

fig_ma30.add_trace(go.Scatter(x=filtered_df1['Date'], y=filtered_df1['MA30'], mode='lines', name=f'{symbol1} 30-day MA'))
fig_ma30.add_trace(go.Scatter(x=filtered_df2['Date'], y=filtered_df2['MA30'], mode='lines', name=f'{symbol2} 30-day MA'))
fig_ma30.add_trace(go.Scatter(x=filtered_df3['Date'], y=filtered_df3['MA30'], mode='lines', name=f'{symbol3} 30-day MA'))

fig_ma30.update_layout(title='30-day Moving Average Over Time', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma30)

# Create combined plot for 90-day moving averages
fig_ma90 = go.Figure()

fig_ma90.add_trace(go.Scatter(x=filtered_df1['Date'], y=filtered_df1['MA90'], mode='lines', name=f'{symbol1} 90-day MA'))
fig_ma90.add_trace(go.Scatter(x=filtered_df2['Date'], y=filtered_df2['MA90'], mode='lines', name=f'{symbol2} 90-day MA'))
fig_ma90.add_trace(go.Scatter(x=filtered_df3['Date'], y=filtered_df3['MA90'], mode='lines', name=f'{symbol3} 90-day MA'))

fig_ma90.update_layout(title='90-day Moving Average Over Time', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma90)

# Daily Change Analysis
st.header("Daily Change Analysis")
selected_symbols_daily_change = st.multiselect("Select Symbols for Daily Change Plot", symbols, default=symbols)
filtered_daily_change_df = combined_df[combined_df['Symbol'].isin(selected_symbols_daily_change)]
fig_daily_change = px.box(filtered_daily_change_df, x='Symbol', y='Daily Change', title='Daily Change in Price (Close - Open) by Stock', color='Symbol', color_discrete_sequence=px.colors.qualitative.Alphabet)
st.plotly_chart(fig_daily_change)

# Optional plots
show_volatility = st.sidebar.checkbox("Show Volatility Plot")
show_scatter = st.sidebar.checkbox("Show Volume vs. Closing Price Plot")
show_rsi = st.sidebar.checkbox("Show RSI Plot")
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands Plot")
show_cumulative_return = st.sidebar.checkbox("Show Cumulative Returns Plot")
show_drawdown = st.sidebar.checkbox("Show Drawdown Plot")

if show_volatility:
    # Sidebar for selecting symbols for the volatility plot
    selected_symbols_volatility = st.sidebar.multiselect("Select Symbols for Volatility Plot", symbols)

    # Filter data based on selected symbols
    filtered_volatility_df = combined_df[combined_df['Symbol'].isin(selected_symbols_volatility)]

    # Create plot for volatility
    fig_volatility = px.line(filtered_volatility_df, x='Date', y='Volatility', color='Symbol', 
                             title='30-Day Rolling Volatility of Tech Stocks')
    st.plotly_chart(fig_volatility)

if show_scatter:
    # Sidebar for selecting symbols for the scatter plot
    selected_symbols_scatter = st.sidebar.multiselect("Select Symbols for Volume vs. Closing Price Plot", symbols)

    # Filter data based on selected symbols
    filtered_scatter_df = combined_df[combined_df['Symbol'].isin(selected_symbols_scatter)]

    # Scatter plot of Volume vs. Closing Price
    fig_scatter = px.scatter(filtered_scatter_df, x='Close', y='Volume', color='Symbol', 
                             title='Volume vs. Closing Price of Stocks')
    fig_scatter.update_layout(xaxis_title='Close Price', yaxis_title='Volume')
    st.plotly_chart(fig_scatter)

if show_rsi:
    # Sidebar for selecting symbols for the RSI plot
    selected_symbols_rsi = st.sidebar.multiselect("Select Symbols for RSI Plot", symbols)

    # Filter data based on selected symbols
    filtered_rsi_df = combined_df[combined_df['Symbol'].isin(selected_symbols_rsi)]

    # Plotting RSI
    fig_rsi = px.line(filtered_rsi_df, x='Date', y='RSI', color='Symbol', 
                      title='Relative Strength Index (RSI) of Tech Stocks')
    st.plotly_chart(fig_rsi)

if show_bollinger:
    # Sidebar for selecting symbols for the Bollinger Bands plot
    selected_symbols_bollinger = st.sidebar.multiselect("Select Symbols for Bollinger Bands Plot", symbols)

    # Filter data based on selected symbols
    filtered_bollinger_df = combined_df[combined_df['Symbol'].isin(selected_symbols_bollinger)]

    # Plotting Bollinger Bands
    fig_bollinger = go.Figure()
    for symbol in filtered_bollinger_df['Symbol'].unique():
        symbol_data = filtered_bollinger_df[filtered_bollinger_df['Symbol'] == symbol]
        fig_bollinger.add_trace(go.Scatter(x=symbol_data['Date'], y=symbol_data['Bollinger_Middle'], name=f'{symbol} Bollinger Middle'))
        fig_bollinger.add_trace(go.Scatter(x=symbol_data['Date'], y=symbol_data['Bollinger_Upper'], name=f'{symbol} Bollinger Upper', line=dict(dash='dash')))
        fig_bollinger.add_trace(go.Scatter(x=symbol_data['Date'], y=symbol_data['Bollinger_Lower'], name=f'{symbol} Bollinger Lower', line=dict(dash='dash')))
    fig_bollinger.update_layout(title='Bollinger Bands for Tech Stocks', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_bollinger)

if show_cumulative_return:
    # Sidebar for selecting symbols for the cumulative returns plot
    selected_symbols_cumulative_return = st.sidebar.multiselect("Select Symbols for Cumulative Returns Plot", symbols)

    # Filter data based on selected symbols
    filtered_cumulative_return_df = combined_df[combined_df['Symbol'].isin(selected_symbols_cumulative_return)]

    # Plotting Cumulative Returns
    fig_cumulative_return = px.line(filtered_cumulative_return_df, x='Date', y='Cumulative Return', color='Symbol', 
                                    title='Cumulative Returns of Tech Stocks')
    st.plotly_chart(fig_cumulative_return)

if show_drawdown:
    # Sidebar for selecting symbols for the drawdown plot
    selected_symbols_drawdown = st.sidebar.multiselect("Select Symbols for Drawdown Plot", symbols)

    # Filter data based on selected symbols
    filtered_drawdown_df = combined_df[combined_df['Symbol'].isin(selected_symbols_drawdown)]

    # Plotting Drawdowns
    fig_drawdown = px.line(filtered_drawdown_df, x='Date', y='Drawdown', color='Symbol', 
                           title='Drawdowns of Tech Stocks')
    st.plotly_chart(fig_drawdown)

# Display descriptive statistics for all three stocks
st.header("Descriptive Statistics")
st.subheader(f'{symbol1} Statistics')
st.write(filtered_df1.describe())

st.subheader(f'{symbol2} Statistics')
st.write(filtered_df2.describe())

st.subheader(f'{symbol3} Statistics')
st.write(filtered_df3.describe())

