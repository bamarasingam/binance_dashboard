#Libraries
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go
from binance.client import Client

#Functions

#Function to load in data from Binance
def load_data(symbol, interval, lookback):
    #Initalize Binance client
    client = Client()

    #Fetch candlestick data
    candles = client.get_historical_klines(symbol, interval, lookback)

    #Create df
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    df_original = df

    #Convert to datetime
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')

    #Keep OHLCV + time
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

    #Convert values to float (other than time)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    #Format time as string value
    df['time'] = df['time'].dt.strftime('%Y-%m-%d')

    #Calculate indicators
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)

    return df

#Function to get emoji based on returns value
def get_returns_emoji(ret_val):
    emoji = ":white_check_mark:"
    if ret_val < 0:
        emoji = ":red_circle:"
    return emoji

#Function to get emoji based on ema value
def get_ema_emoji(ltp,ema):
    emoji = ":white_check_mark:"
    if ltp < ema:
        emoji = ":red_circle:"
    return emoji

#Function to get emoji based on rsi value
def get_rsi_emoji(rsi):
    emoji = ":red_circle:"
    if 30 < rsi < 70:
        emoji = ":white_check_mark:"
    return emoji

#Function to get emoji based on adx value
def get_adx_emoji(adx):
    emoji = ":red_circle:"
    if adx > 25:
        emoji = ":white_check_mark:"
    return emoji

#Function to create chart
def create_chart(df, symbol):
    candlestick_chart = go.Figure(data=[go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'])])
    ema20 = go.Scatter(x = df.EMA_20.index,y = df.EMA_20.values,name = 'EMA20')
    ema200 = go.Scatter(x = df.EMA_200.index,y = df.EMA_200.values,name = 'EMA200')
    # Create the candlestick chart
    candlestick_chart.update_layout(title=f'{symbol} Historical Candlestick Chart',
                                        xaxis_title='Date',
                                        yaxis_title='Price',
                                        xaxis_rangeslider_visible=True)
    candlestick_chart.add_trace(ema20)
    candlestick_chart.add_trace(ema200)
    return candlestick_chart

#Streamlit

#Centered title
st.markdown("<h2 style='text-align: center;'>Crypto Technical Analysis Dashboard</h2>", unsafe_allow_html=True)

#Sidebar Components
symbol = st.sidebar.text_input("Crypto Symbol (e.g. BTCUSDT)", "BTCUSDT")
interval = st.sidebar.selectbox("Interval", 
                                ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'))
lookback = st.sidebar.selectbox("Lookback period", 
                                ('1 day ago UTC', '1 week ago UTC', '1 month ago UTC', '3 months ago UTC', '6 months ago UTC', '1 year ago UTC'))
show_data = st.sidebar.checkbox(label="Show Data")
show_chart = st.sidebar.checkbox(label="Show Chart")

df = load_data(symbol, interval, lookback)
reversed_df = df.iloc[::-1]
row1_val = reversed_df.iloc[0]['close']
ema20_val = reversed_df.iloc[0]['EMA_20']
ema200_val = reversed_df.iloc[0]['EMA_200']
rsi_val = reversed_df.iloc[0]['RSI_14']
adx = reversed_df.iloc[0]['ADX_14']
dmp = reversed_df.iloc[0]['DMP_14']
dmn = reversed_df.iloc[0]['DMN_14']

row20_val = reversed_df.iloc[20]['close'] if len(reversed_df) > 20 else row1_val
row60_val = reversed_df.iloc[60]['close'] if len(reversed_df) > 60 else row1_val
row120_val = reversed_df.iloc[120]['close'] if len(reversed_df) > 120 else row1_val
row240_val = reversed_df.iloc[240]['close'] if len(reversed_df) > 240 else row1_val

#Return Percentage Calculation
day20_ret_percent = (row1_val - row20_val)/row20_val * 100
day20_ret_val = (row1_val - row20_val)
day60_ret_percent = (row1_val - row60_val)/row60_val * 100
day60_ret_val = (row1_val - row60_val)
day120_ret_percent = (row1_val - row120_val)/row120_val * 100
day120_ret_val = (row1_val - row120_val)
day240_ret_percent = (row1_val - row240_val)/row240_val * 100
day240_ret_val = (row1_val - row240_val)

#Column wise Display
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Returns")
    st.markdown(f"- 1  MONTH : {round(day20_ret_percent,2)}% {get_returns_emoji(round(day20_ret_percent,2))}")
    st.markdown(f"- 3  MONTHS : {round(day60_ret_percent,2)}% {get_returns_emoji(round(day60_ret_percent,2))}")
    st.markdown(f"- 6  MONTHS : {round(day120_ret_percent,2)}% {get_returns_emoji(round(day120_ret_percent,2))}")
    st.markdown(f"- 12 MONTHS : {round(day240_ret_percent,2)}% {get_returns_emoji(round(day240_ret_percent,2))}")
with col2:
    st.subheader("Momentum")
    st.markdown(f"- LTP : {round(row1_val,2)}") 
    st.markdown(f"- EMA20 : {round(ema20_val,2)} {get_ema_emoji(round(row1_val,2),round(ema20_val,2))}") 
    st.markdown(f"- EMA200 : {round(ema200_val,2)} {get_ema_emoji(round(row1_val,2),round(ema200_val,2))}") 
    st.markdown(f"- RSI : {round(rsi_val,2)} {get_rsi_emoji(round(rsi_val,2))}") 
with col3:
    st.subheader("Trend Strength")
    st.markdown(f"- ADX : {round(adx,2)} {get_adx_emoji(round(adx,2))}") 
    st.markdown(f"- DMP : {round(dmp,2)} ") 
    st.markdown(f"- DMN : {round(dmn,2)} ") 

if show_data:
    st.write(reversed_df)

if show_chart:
    st.plotly_chart(create_chart(df, symbol))