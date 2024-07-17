    #Centered title
    st.markdown("<h2 style='text-align: center;'>Predictions Using Logistic Regression</h2>", unsafe_allow_html=True)

    #Create binary target variable
    df['long_or_short'] = (df['close'] > df['open']).astype(int)

    #Display the DataFrame with calculated lags and target
    if show_lag_data:
        st.markdown("<h3 style='text-align: center;'>Data with Calculated Lags and Target</h3>", unsafe_allow_html=True)
        st.write(df[['open', 'close', 'volume', 'returns', 'long_or_short'] + lagnames].sort_index(ascending=False))

    # Build model
    X = df[lagnames]
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

    # Display the table in Streamlit DELETE
    st.subheader("Comparison of Actual vs Predicted Values")
    st.dataframe(comparison_df)

    st.write("Class distribution:")
    st.write(df['long_or_short'].value_counts(normalize=True))