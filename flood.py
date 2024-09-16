import streamlit as st
import pandas as pd
import numpy as np
import smtplib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from keras import saving

import requests
import seaborn as sns

st.set_page_config(page_title="Flood Prediction",page_icon="🌧",layout="wide")

# Load the model with the custom deserialization function
model = saving.load_model(r'LSTM.keras')


markov_df = pd.read_csv(r'transition_matrix.csv')
transition_matrix = markov_df.values

def send_mail(message):
    email = "sundayabraham81@gmail.com"
    reciever = "sundayabraham357@gmail.com"

    subject = "Flood prediction"
    #message = "Flood predicted in 5 hours"

    text = f"subject: {subject} \n\n{message}"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    server.login(email, "riai arnl gvwt dxcc")
    server.sendmail(email, reciever, text)

def get_date():
    
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def markov_prediction(input_value):
    if input_value == 0:
        input_value = [1,0,0]
    elif input_value == 1:
        input_value = [0,1,0]
    elif input_value == 2:
        input_value = [0,0,1]
        
    initial_state = np.array(input_value)
    state_distribution = initial_state
    output_value = np.dot(state_distribution, transition_matrix)
    return output_value


def classify(water_level):
    if water_level < 84:
        return 0
    elif water_level < 105:
        return 1
    elif water_level > 105:
        return 2


    

# Nexmo (Vonage) API credentials
NEXMO_API_KEY = 'your_nexmo_api_key'
NEXMO_API_SECRET = 'your_nexmo_api_secret'
NEXMO_PHONE_NUMBER = 'your_nexmo_phone_number'
USER_PHONE_NUMBER = 'user_phone_number'

# Initialize input data storage
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = []
    st.session_state['raw_data'] = []

# Function to preprocess data
def preprocess_data(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data[-12:], columns=['datetime', 'water_level'])
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Create lag features for LSTM
    for lag in range(12):
        df[f'col_{lag}'] = df['water_level'].shift(lag)

    # Extract date features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['year'] = df['datetime'].dt.year
    df['bidaily'] = df['datetime'].dt.dayofweek
    df['minute'] = df['datetime'].dt.minute
    

    

    df.fillna(0, inplace = True)
    
    return df

# Function to send SMS alert
def send_sms(message):
    url = 'https://rest.nexmo.com/sms/json'
    payload = {
        'api_key': NEXMO_API_KEY,
        'api_secret': NEXMO_API_SECRET,
        'to': USER_PHONE_NUMBER,
        'from': NEXMO_PHONE_NUMBER,
        'text': message
    }
    response = requests.post(url, data=payload)
    return response 


def forecast_next_twelve(df, lstm_model, transition_mat):
    # Make a copy of the original dataframe to avoid modifying it
    forecast_df = df.copy()

    for _ in range(12):
        # Extract features for LSTM
        X_lstm = forecast_df[['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'hour', 'dayofweek', 'month', 'quarter', 'year', 'bidaily']].iloc[-1].values.reshape(1,18,1)
        # Extract the current water level for Markov Chain
        X_markov = forecast_df['water_level'].iloc[-1]


        if X_markov == 0:
            markov_mat = np.array([1, 0, 0])
        elif X_markov == 1:
            markov_mat = np.array([0, 1, 0])
        elif X_markov == 2:
            markov_mat = np.array([0, 0, 1])

        # Predict probabilities with both models
        lstm_probs = model.predict(X_lstm)
        markov_probs = np.dot(transition_mat, markov_mat)

        # Ensemble the probabilities
        final_probs = 0.8 * lstm_probs + 0.2 * markov_probs
        next_class = np.argmax(final_probs)

        # Generate next datetime
        next_datetime = forecast_df['datetime'].iloc[-1] + pd.Timedelta(hours=1)

        # Append the predicted value to the forecast dataframe
        forecast_df = pd.concat([forecast_df, pd.DataFrame({'datetime': next_datetime, 'water_level': next_class}, index = [1])], ignore_index=True, axis=0)#forecast_df.append({'datetime': next_datetime, 'water_level': next_class}, ignore_index=True)

        # Update lag features for the next iteration
        for col in range(12):
            forecast_df[f'col_{col}'] = forecast_df['water_level'].shift(col)

        # Update date features for the next iteration
        forecast_df['hour'] = forecast_df['datetime'].dt.hour
        forecast_df['dayofweek'] = forecast_df['datetime'].dt.dayofweek
        forecast_df['month'] = forecast_df['datetime'].dt.month
        forecast_df['quarter'] = forecast_df['datetime'].dt.quarter
        forecast_df['year'] = forecast_df['datetime'].dt.year
        forecast_df['bidaily'] = forecast_df['datetime'].dt.dayofweek
        forecast_df['minute'] = forecast_df['datetime'].dt.minute
        

        # Drop rows with missing values after feature update
        forecast_df.fillna(0, inplace = True)
    table_df = forecast_df[['datetime', 'water_level']].tail(12).copy()
    table_df.rename(columns={'datetime': 'Date', 'water_level': 'Flood Prediction'}, inplace=True)
    prediction_list = table_df['Flood Prediction']
    mapping = {0: "NO FLOOD", 1: "ALMOST FLOOODED", 2: "FLOOD"}
    table_df['Flood Prediction'] = table_df['Flood Prediction'].map(mapping)
    table_df.index = [np.arange(1, len(table_df)+1)]

    return table_df, prediction_list


def floodpred():
    # App title and description
    st.write(
                        """
                        <div style="background-color: #4682B4; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 46px;">
                            Welcome to Flood Prediction Web App
                        </div>
                        """,
                        unsafe_allow_html=True
                        )
    st.write("")
    st.write("")

    st.title('Get Predictions on the Possibility of Flood')

    st.write("")
    st.write("")


    # Input form
    st.subheader('Input Water Level Data')
    st.text_input('Datetime (YYYY-MM-DD HH:MM)', value=datetime.now().strftime('%Y-%m-%d %H:%M'))
    water_level_input = st.number_input('Water Level', min_value=0.0, max_value=200.0)

    if st.button('Submit Data'):
        datetime_input = get_date() 
        st.session_state['raw_data'].append(water_level_input)
        st.session_state['input_data'].append([datetime_input, classify(water_level_input)])
        st.success('Data submitted!')

    # Check if enough data points are collected
    if len(st.session_state['input_data']) >= 12:
        # Preprocess the data
        df = preprocess_data(st.session_state['input_data'])
        
        # Make predictions
        X_lstm = df[['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11',  'hour', 'dayofweek', 'month', 'quarter', 'year', 'bidaily']]
        X_markov = df['water_level']
        
        df['lstm_probs'] = df['water_level'].apply(markov_prediction)#[[0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7]]#lstm_model.predict(X_lstm)
        df['markov_probs'] = df['water_level'].apply(markov_prediction)#[[0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7]]#markov_prediction(X_markov)
        
        
        # Ensemble predictions
        # Assuming df['lstm_probs'] and df['markov_probs'] are Series of lists
        df['final_probs'] = df.apply(lambda row: [(0.5 * lstm + 0.5 * markov) for lstm, markov in zip(row['lstm_probs'], row['markov_probs'])], axis=1)#(0.5 * list(df['lstm_probs'])) + (0.5 * list(df['markov_probs']))
        df['final_preds'] = df['final_probs'].apply(lambda x: np.argmax(x))
        
        # Display predictions
        #st.subheader('Prediction for next seven hours')
        #st.write(f'Predicted class for next hour: {df['final_preds'].iloc[-1]}')
        #if df['final_preds'].iloc[-1] == 2:
        #    st.error('Flood predicted!')
        #    send_sms('Flood alert! Take necessary precautions.')"""
        col1, col2 = st.columns(2)
        with col2:
            
            st.header('Forecast of Next Twelve Hours')
            st.write("")
            st.write("")
            table, pred_list = forecast_next_twelve(df, 5, transition_matrix)
            st.table(table)
            max, argmax = pred_list.max(), pred_list.argmax()
            if max == 2 & argmax>0:
                send_mail(f'Flood predicted in {argmax+1}hours time!!!')
            elif max == 2 & argmax==0:
                send_mail(f'Flood predicted in an hour time!!!')


        with col1:  
            # Display past 7 days chart
            st.header('Past 12 Hours Water Levels Chart')
            st.write("")
            st.write("")

            fig = plt.figure(figsize=(10, 6))
            #print(f'df: {df.head(8)}')
            sns.barplot(pd.DataFrame({'Time':df['minute'], 'Water Level': st.session_state['raw_data'][-12:]}),
                        x = 'Time', 
                        y = 'Water Level')
            plt.title('Water Level Plot for Past 12 Hours')
            st.pyplot(fig)

def navigate_to_page(page):
    st.session_state.page = page
    
def homepage():
    log_in =0

    username_real = 'flood prediction'
    password_real = 'computerengineering'
    #st.title('WELCOME TO FLOOD PREDICTION LOGIN PAGE')
    
    col1, col2, col3 = st.columns(3)
    with col2:
        st.write(
                            """
                            <div style="background-color: #4682B4; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 24px;">
                                WELCOME TO FLOOD PREDICTION LOGIN PAGE
                            </div>
                            """,
                            unsafe_allow_html=True
                            )

        st.title("")
        st.title("")

        username = st.text_input('username', max_chars=30)

        #st.title("")

        password = st.text_input('password', max_chars = 20, type = 'password')
        #st.title("")

    def password_check():
        if log_in:
            if (username == username_real) & (password == password_real):
                navigate_to_page('floodpred')
            
            else:
                st.write("wrong username or password")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col3:
        log_in = st.button('Log in', help = 'double click to login', on_click=password_check)


if 'page' not in st.session_state:
    st.session_state.page = 'homepage'  # Set initial page

# Conditional rendering based on the current page
if st.session_state.page == 'homepage':
    homepage()
elif st.session_state.page == 'floodpred':
    floodpred()
    

# Run the Streamlit app
#if __name__ == '__main__':
#    st.run()


 

# Example usage
# Assuming df is your initial dataframe with the past 7 values
# and lstm_model and markov_model are already loaded
#forecasted_values = forecast_next_seven(df)#, lstm_model)
#print(forecasted_values)
