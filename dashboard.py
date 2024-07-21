#Libraries
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from binance.client import Client
from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Libraries below may be used for anothr time
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.impute import SimpleImputer

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
    #Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f'{symbol} {chart_type} Chart', 'Volume'),
                        row_heights=[0.7, 0.3])

    #Add price chart
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
    
    #Add EMA lines
    fig.add_trace(
        go.Scatter(x=df.index, y=df.EMA_20.values, name='EMA20', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df.EMA_200.values, name='EMA200', line=dict(color='red')),
        row=1, col=1
    )
    
    #Add volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume'),
        row=2, col=1
    )
    
    #Update layout
    fig.update_layout(
        title_text=f'{symbol} Historical Data',
        xaxis_rangeslider_visible=False,
        height=800,  # Increase overall height of the figure
        showlegend=True
    )

    #Update y-axis
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
        row=2, col=1  #Add range selector to bottom subplot
    )
    
    return fig

#Function for lags using returns
def lagit(df, lags):
    names = []
    for i in range(1, lags+1):
        df['Lag_' + str(i)] = df['returns'].shift(i)
        names.append('Lag_' + str(i))
    return names

#Updated lag function which includes lag direction
def lagit_dir(df, lags):
    names = []
    for i in range(1, lags+1):
        df['Lag_'+str(i)] = df['returns'].shift(i)
        df['Lag_'+str(i)+'_dir'] = [1 if j>0 else 0 for j in df['Lag_'+str(i)]]
        names.append('Lag_'+str(i)+'_dir')
    return names

#Colors for trigger words
def color_text(val):
    if val in ['Increase', 'Buy']:
        return 'color: green'
    elif val in ['Decrease', 'Sell']:
        return 'color: red'
    return ''


#---------------------------------------------------Streamlit-------------------------------------------------

#Create tabs for dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data/Analytics", "Lin Regression", "Log Regression", "RF Regressor", "RF Classification"])

#Data/Analytics Tab
with tab1:
    #Centered title
    st.markdown("<h2 style='text-align: center;'>Technical Analysis Dashboard</h2>", unsafe_allow_html=True) #Center title

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

    st.sidebar.info("Note: The latest row is within a timeframe which has not completed. The close price is the price at the time at which you pulled the data for this most recent row.")

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
    
    #Display chart 
    st.plotly_chart(create_chart(df, symbol, chart_type), use_container_width=True)
    #Display dataframe
    st.write(reversed_df)
        
#Linear Regression Tab
with tab2:
    #Centered title
    st.markdown("<h2 style='text-align: center;'>Predictions Using Linear Regression</h2>", unsafe_allow_html=True)

    #Create returns column
    df['returns'] = np.log(df.close.pct_change() + 1)

    #Call function with associated amount of lags
    lagnames = lagit(df, 5)
    df.dropna(inplace=True)

    #Display the DataFrame with calculated lags
    st.subheader('Data with Calculated Lags')
    st.write(df[['open', 'close', 'volume', 'returns'] + lagnames].sort_index(ascending=False))

    #Build, train and test model
    X = df[['open'] + lagnames]
    y = df['close']

    lr = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    #Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Actual Close Price', color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted Close Price', color='red')
    ax.set_title('Actual vs Predicted Close Price of Test Sample (Most Recent 1/5 of Data)')
    ax.legend()
    plt.xticks(rotation=45)

    #Display the plot in Streamlit
    st.pyplot(fig)

    # Calculate metrics
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    #Display metrics
    metrics_df = pd.DataFrame({'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R-squared Score'],'Value': [mse, rmse, mae, r2]})
    st.table(metrics_df.style.format({'Value': '{:.6f}'}))

    #Prediction for the latest data point
    latest_data = X.iloc[-1].values.reshape(1, -1)
    latest_prediction = lr.predict(latest_data)[0]

    st.subheader(f"Prediction for Upcoming Close Price: {latest_prediction}")

    #Create DataFrame with actual and predicted values, as well as actual and predicted actions
    results_df = pd.DataFrame({
        'Open': X_test['open'],
        'Actual Close': y_test,
        'Predicted Close': y_pred,
    })
    results_df['Actual Price Movement'] = np.where(results_df['Actual Close'] > results_df['Open'], 'Increase', 'Decrease')
    results_df['Predicted Price Action'] = np.where(results_df['Predicted Close'] > results_df['Open'], 'Buy', 'Sell')

    #Round the values to 2 decimal places
    results_df = results_df.round(2).sort_index(ascending=False)

    #Style dataframe
    styled_df = results_df.style.applymap(color_text, subset=['Actual Price Movement', 'Predicted Price Action'])

    st.dataframe(styled_df)

    correct_predictions = (results_df['Actual Price Movement'] == 'Increase') & (results_df['Predicted Price Action'] == 'Buy') | \
                          (results_df['Actual Price Movement'] == 'Decrease') & (results_df['Predicted Price Action'] == 'Sell')
    accuracy = correct_predictions.mean()

    st.write(f"Trading Signal Accuracy: {accuracy:.2%}")

#Logistic Regression Tab
with tab3:
    #Centered title
    st.markdown("<h2 style='text-align: center;'>Predictions Using Logistic Regression</h2>", unsafe_allow_html=True)

    #Set up returns and direction
    df['returns'] = np.log(df.close.pct_change() + 1)
    df['direction'] = [1 if i>0 else 0 for i in df.returns]

    dirnames = lagit_dir(df, 5)

    df.dropna(inplace=True)

    X = df[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']+dirnames]
    y = df['direction']

    log_reg = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    #Display DataFrame
    st.subheader('Data with Calculated Lags and Direction')
    st.dataframe(df[['open', 'close', 'volume', 'returns', 'direction'] + ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5'] + dirnames].sort_index(ascending=False))

    #Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

    #Classification Report
    st.subheader('Classification Report')
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

    #Prediction for latest time
    latest_data = X.iloc[-1].values.reshape(1, -1)
    latest_prediction = log_reg.predict(latest_data)[0]
    latest_probability = log_reg.predict_proba(latest_data)[0][1]

    if latest_prediction == 1:
        st.subheader(f"Prediction for Upcoming Close Price: Long")
    else:
        st.subheader(f"Prediction for Upcoming Close Price: Short")

    #st.write(f'Probability of price going up: {latest_probability:.2f}')

    #Create dataframe which shows y_test and y_pred
    results_df = pd.DataFrame({
    'Actual Price Movement': y_test,
    'Predicted Action': y_pred
    },)
    results_df['Open'] = df.loc[results_df.index, 'open']
    results_df['Close'] = df.loc[results_df.index, 'close']
    results_df['Price Change %'] = ((results_df['Close'] - results_df['Open']) / results_df['Open'] * 100).round(2)


    column_order = ['Open', 'Close', 'Actual Price Movement', 'Predicted Action', 'Price Change %']
    results_df = results_df[column_order]

    #Convert binary values to text
    results_df['Actual Price Movement'] = results_df['Actual Price Movement'].map({1: 'Price Up', 0: 'Price Down'})
    results_df['Predicted Action'] = results_df['Predicted Action'].map({1: 'Long', 0: 'Short'})

    #Sort the DataFrame by date in descending order
    results_df = results_df.sort_index(ascending=False)

    #Define color function
    def color_text(val):
        if val in ['Price Up', 'Long']:
            return 'color: green'
        elif val in ['Price Down', 'Short']:
            return 'color: red'
        return ''

    #Style dataframe with above function
    styled_df = results_df.style.applymap(color_text)
    #Apply color to price change percentage
    styled_df = styled_df.applymap(lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '', subset=['Price Change %'])  

    #Display the styled DataFrame
    st.dataframe(styled_df)

    #Model Accuracy
    correct_predictions = ((results_df['Actual Price Movement'] == 'Price Up') & (results_df['Predicted Action'] == 'Long')) | \
                          ((results_df['Actual Price Movement'] == 'Price Down') & (results_df['Predicted Action'] == 'Short'))
    accuracy = correct_predictions.mean()

    st.write(f"Trading Signal Accuracy: {accuracy:.2%}")

with tab4:
    #Centered title
    st.markdown("<h2 style='text-align: center;'>Predictions Using Random Forest Regressor</h2>", unsafe_allow_html=True)

    X = df[['open', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']+dirnames]
    y = df['close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    #Display DataFrame
    st.subheader('Data with Calculated Lags')
    st.dataframe(df[['open', 'close', 'volume', 'returns'] + ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5'] + dirnames].sort_index(ascending=False))

    #Create and display plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Actual Close Price', color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted Close Price', color='red')
    ax.set_title('Actual vs Predicted Close Price of Test Sample (Most Recent 1/5 of Data)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    #Calculate metrics
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    #Display metrics
    metrics_df = pd.DataFrame({'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R-squared Score'],'Value': [mse, rmse, mae, r2]})
    st.table(metrics_df.style.format({'Value': '{:.6f}'}))

    #Prediction for the latest data point
    latest_data = X.iloc[-1].values.reshape(1, -1)
    latest_prediction = rf.predict(latest_data)[0]

    st.subheader(f"Prediction for Upcoming Close Price: {latest_prediction}")

    #Create DataFrame with actual and predicted values, as well as actual and predicted actions
    results_df = pd.DataFrame({
        'Open': X_test['open'],
        'Actual Close': y_test,
        'Predicted Close': y_pred,
    })
    results_df['Actual Price Movement'] = np.where(results_df['Actual Close'] > results_df['Open'], 'Increase', 'Decrease')
    results_df['Predicted Price Action'] = np.where(results_df['Predicted Close'] > results_df['Open'], 'Buy', 'Sell')

    #Round the values to 2 decimal places
    results_df = results_df.round(2).sort_index(ascending=False)

    #Style dataframe
    styled_df = results_df.style.applymap(color_text, subset=['Actual Price Movement', 'Predicted Price Action'])

    st.dataframe(styled_df)

    correct_predictions = (results_df['Actual Price Movement'] == 'Increase') & (results_df['Predicted Price Action'] == 'Buy') | \
                          (results_df['Actual Price Movement'] == 'Decrease') & (results_df['Predicted Price Action'] == 'Sell')
    accuracy = correct_predictions.mean()

    st.write(f"Trading Signal Accuracy: {accuracy:.2%}")

with tab5:
    st.markdown("<h2 style='text-align: center;'>Predictions Using Random Forest Classification</h2>", unsafe_allow_html=True)

    #Set up returns and direction
    df['returns'] = np.log(df.close.pct_change() + 1)
    df['direction'] = [1 if i>0 else 0 for i in df.returns]

    dirnames = lagit_dir(df, 5)

    df.dropna(inplace=True)

    X = df[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']+dirnames]
    y = df['direction']

    rfc = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    #Display DataFrame
    st.subheader('Data with Calculated Lags and Direction')
    st.dataframe(df[['open', 'close', 'volume', 'returns', 'direction'] + ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5'] + dirnames].sort_index(ascending=False))

    #Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

    #Classification Report
    st.subheader('Classification Report')
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

    #Prediction for latest time
    latest_data = X.iloc[-1].values.reshape(1, -1)
    latest_prediction = rfc.predict(latest_data)[0]
    latest_probability = rfc.predict_proba(latest_data)[0][1]

    if latest_prediction == 1:
        st.subheader(f"Prediction for Upcoming Close Price: Long")
    else:
        st.subheader(f"Prediction for Upcoming Close Price: Short")

    #st.write(f'Probability of price going up: {latest_probability:.2f}')

    #Create dataframe which shows y_test and y_pred
    results_df = pd.DataFrame({
    'Actual Price Movement': y_test,
    'Predicted Action': y_pred
    },)
    results_df['Open'] = df.loc[results_df.index, 'open']
    results_df['Close'] = df.loc[results_df.index, 'close']
    results_df['Price Change %'] = ((results_df['Close'] - results_df['Open']) / results_df['Open'] * 100).round(2)


    column_order = ['Open', 'Close', 'Actual Price Movement', 'Predicted Action', 'Price Change %']
    results_df = results_df[column_order]

    # Convert binary values to text
    results_df['Actual Price Movement'] = results_df['Actual Price Movement'].map({1: 'Price Up', 0: 'Price Down'})
    results_df['Predicted Action'] = results_df['Predicted Action'].map({1: 'Long', 0: 'Short'})

    #Sort the DataFrame by date in descending order
    results_df = results_df.sort_index(ascending=False)

    #Define color function
    def color_text(val):
        if val in ['Price Up', 'Long']:
            return 'color: green'
        elif val in ['Price Down', 'Short']:
            return 'color: red'
        return ''

    #Style dataframe with above function
    styled_df = results_df.style.applymap(color_text)
    #Apply color to price change percentage
    styled_df = styled_df.applymap(lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '', subset=['Price Change %'])  

    #Display the styled DataFrame
    st.dataframe(styled_df)

    #Model Accuracy
    correct_predictions = ((results_df['Actual Price Movement'] == 'Price Up') & (results_df['Predicted Action'] == 'Long')) | \
                          ((results_df['Actual Price Movement'] == 'Price Down') & (results_df['Predicted Action'] == 'Short'))
    accuracy = correct_predictions.mean()

    st.write(f"Trading Signal Accuracy: {accuracy:.2%}")