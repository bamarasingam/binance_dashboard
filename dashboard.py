#Libraries
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from binance.client import Client
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
#---------------------------------------------------Functions-------------------------------------------------

#Function to load in data from Binance
def load_data(symbol, interval, start_date):
    #Initalize Binance client
    client = Client()

    #Convert start_date to ms timestamp
    start_datetime = datetime.combine(start_date, datetime.min.time())
    start_ts = int(start_datetime.timestamp() * 1000)

    #Fetch candlestick data
    candles = client.get_historical_klines(symbol, interval, start_ts)

    #Create df
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    #Convert to datetime
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    #Keep OHLCV + time
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

    #Convert values to float (other than time)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    #Format time as string value
    #df['time'] = df['time'].dt.strftime('%Y-%m-%d')
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    #Calculate indicators
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=14, append=True)
    df.ta.macd(length=20,append = True)

    return df

#Function to get emoji based on returns value
def get_returns_emoji(ret_val):
    return ":white_check_mark:" if ret_val >= 0 else ":red_circle:"

#Function to get emoji based on ema value
def get_ema_emoji(ltp,ema):
    return ":white_check_mark:" if ltp >= ema else ":red_circle:"

#Function to get emoji based on rsi value
def get_rsi_emoji(rsi):
    return ":white_check_mark:" if 30 < rsi < 70 else ":red_circle:"

#Function to get emoji based on adx value
def get_adx_emoji(adx):
    return ":white_check_mark:" if adx > 25 else ":red_circle:"

#Function to create chart
def create_chart(df, symbol, chart_type):
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f'{symbol} {chart_type} Chart', 'Volume'),
                        row_heights=[0.7, 0.3])

    # Add main price chart
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(x=df.index,
                           open=df['open'],
                           high=df['high'],
                           low=df['low'],
                           close=df['close'],
                           name="Price"),
            row=1, col=1
        )
    else:  # Line chart
        fig.add_trace(
            go.Scatter(x=df.index,
                       y=df['close'],
                       mode='lines',
                       name="Price"),
            row=1, col=1
        )
    
    # Add EMA lines
    fig.add_trace(
        go.Scatter(x=df.index, y=df.EMA_20.values, name='EMA20', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df.EMA_200.values, name='EMA200', line=dict(color='red')),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume'),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title_text=f'{symbol} Historical Data',
        xaxis_rangeslider_visible=False,
        height=800,  # Increase overall height of the figure
        showlegend=True
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Update x-axis
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        row=2, col=1  # Add range selector to bottom subplot
    )
    
    return fig

#Function for lags using returns
def lagit(df, lags):
    names = []
    for i in range(1, lags+1):
        df['Lag_' + str(i)] = df['returns'].shift(i)
        names.append('Lag_' + str(i))
    return names

#Function to predict close price for given tiemframe
def predict_close(model, open_price, lags):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[open_price] + lags], columns=['open'] + [f'Lag_{i}' for i in range(1, len(lags)+1)])
    
    # Make the prediction
    predicted_close = model.predict(input_data)[0]
    
    return predicted_close
#---------------------------------------------------Streamlit-------------------------------------------------

#Create tabs for dashboard
tab1, tab2, tab3 = st.tabs(["Data/Analytics", "Linear Regression", "Logistic Regression"])

#Data/Analytics Tab
with tab1:
    #Centered title
    st.markdown("<h2 style='text-align: center;'>Crypto Technical Analysis Dashboard</h2>", unsafe_allow_html=True)

    #Sidebar Components
    symbol = st.sidebar.text_input("Crypto Symbol (ex. BTCUSDT)\n\nMust be a ticker from Binance", "BTCUSDT")
    intervals = ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
    default_index = intervals.index('1d')
    interval = st.sidebar.selectbox("Interval", 
                                    options=intervals,
                                    index=default_index)

    #Date input for start date
    default_date = datetime.now().date() - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", 
                                    value= default_date,
                                    max_value=datetime.now())
    chart_type = st.sidebar.radio("Chart Type", ("Candlestick", "Line"))
    show_chart = st.sidebar.checkbox(label="Show Chart/Volume", value = True)
    show_original_data = st.sidebar.checkbox(label="Show Original Data", value = True)
    show_lag_data = st.sidebar.checkbox(label="Show Data with Lags", value = True)

    df = load_data(symbol, interval, start_date)
    reversed_df = df.iloc[::-1] #Reversed dataframe to be shown in Streamlit
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

    day20_ret_percent = (row1_val - row20_val)/row20_val * 100
    day60_ret_percent = (row1_val - row60_val)/row60_val * 100
    day120_ret_percent = (row1_val - row120_val)/row120_val * 100
    day240_ret_percent = (row1_val - row240_val)/row240_val * 100

    #Displays (column wide)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Returns")
        st.markdown(f"- 1 MONTH : {round(day20_ret_percent,2)}% {get_returns_emoji(round(day20_ret_percent,2))}")
        st.markdown(f"- 3 MONTHS : {round(day60_ret_percent,2)}% {get_returns_emoji(round(day60_ret_percent,2))}")
        st.markdown(f"- 6 MONTHS : {round(day120_ret_percent,2)}% {get_returns_emoji(round(day120_ret_percent,2))}")
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

    if show_chart:
        st.plotly_chart(create_chart(df, symbol, chart_type), use_container_width=True)

    if show_original_data:
        st.write(reversed_df)
        
#Linear Regression Tab
with tab2:
    #Centered title
    st.markdown("<h2 style='text-align: center;'>Predictions Using Linear Regression</h2>", unsafe_allow_html=True)

    # Create returns column
    df['returns'] = np.log(df.close.pct_change() + 1)

    # Call function with associated amount of lags
    lagnames = lagit(df, 5)
    df.dropna(inplace=True)

    # Display the DataFrame with calculated lags
    if show_lag_data:
        st.markdown("<h3 style='text-align: center;'>Data with Calculated Lags</h3>", unsafe_allow_html=True)
        st.write(df[['open', 'close', 'volume', 'returns'] + lagnames].sort_index(ascending=False))

    # Build model
    X = df[['open'] + lagnames]
    y = df['close']

    lr = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Actual Close Price', color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted Close Price', color='red')
    ax.set_title('Actual vs Predicted Close Price of Test Sample (Most Recent 1/5 of Data)')
    ax.legend()
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Calculate metrics
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    # Display metrics
    st.subheader('Model Performance Metrics')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Squared Error", f"{mse:.6f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.6f}")
    col3.metric("Mean Absolute Error", f"{mae:.6f}")
    col4.metric("R-squared Score", f"{r2:.6f}")

    # Get the most recent data point
    latest_data = df.iloc[-1]

    # Extract the required features
    open_price = latest_data['open']
    lags = [latest_data[f'Lag_{i}'] for i in range(1, 6)]

    # Make the prediction
    predicted_close = predict_close(lr, open_price, lags)

    # Display the prediction
    col1, col2 = st.columns(2)
    with col1:
        st.write("Input features used for prediction:")
        st.write(pd.DataFrame({'Feature': ['Open'] + [f'Lag_{i}' for i in range(1, 6)], 'Value': [open_price] + lags}))
        
    with col2:
        st.metric(label="Predicted Close for Timeframe", value=f"${predicted_close:.2f}")

#Logistic Regression Tab
with tab3:
    #Centered title
    st.markdown("<h2 style='text-align: center;'>Predictions Using Logistic Regression</h2>", unsafe_allow_html=True)

    #Create binary target variable
    df['long_or_short'] = (df['close'] > df['open']).astype(int)

    #Display the DataFrame with calculated lags and target
    if show_lag_data:
        st.markdown("<h3 style='text-align: center;'>Data with Calculated Lags and Target</h3>", unsafe_allow_html=True)
        st.write(df[['open', 'close', 'volume', 'returns', 'long_or_short'] + lagnames].sort_index(ascending=False))

    # Build model
    X = df[['open'] + lagnames]
    y = df['long_or_short']

    log_reg = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

    #Calculate the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    #Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Display the confusion matrix in Streamlit
    st.pyplot(plt)

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Display metrics
    st.subheader('Model Performance Metrics')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    # Function to predict direction
    def predict_direction(model, open_price, lags):
        input_data = pd.DataFrame([[open_price] + lags], columns=['open'] + [f'Lag_{i}' for i in range(1, len(lags)+1)])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        return prediction, probability

    # Get the most recent data point
    latest_data = df.iloc[-1]

    # Extract the required features
    open_price = latest_data['open']
    lags = [latest_data[f'Lag_{i}'] for i in range(1, 6)]

    # Make the prediction
    predicted_direction, predicted_probability = predict_direction(log_reg, open_price, lags)

    # Display the prediction
    col1, col2 = st.columns(2)
    with col1:
        st.write("Input features used for prediction:")
        st.write(pd.DataFrame({'Feature': ['Open'] + [f'Lag_{i}' for i in range(1, 6)], 'Value': [open_price] + lags}))
        
    with col2:
        direction = "Long (Price will go up)" if predicted_direction == 1 else "Short (Price will go down)"
        st.metric(label="Predicted Direction for Timeframe", value=direction)
        st.metric(label="Probability of Price Increase", value=f"{predicted_probability:.2%}")

    # Create a DataFrame with y_test and y_pred
    comparison_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Predicted Probability': y_pred_proba
    }, index=y_test.index)

    # Add a column to show if the prediction was correct
    comparison_df['Correct'] = comparison_df['Actual'] == comparison_df['Predicted']

    # Display the table in Streamlit
    st.subheader("Comparison of Actual vs Predicted Values")
    st.dataframe(comparison_df)