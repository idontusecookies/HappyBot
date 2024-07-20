import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import random
import plotly.express as px

# Load the trained model and vectorizer
with open('emotion_clf.pkl', 'rb') as f:
    emotion_clf = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define a function to predict emotion and probabilities
def predict_emotion(user_input):
    X = vectorizer.transform([user_input])
    emotion = emotion_clf.predict(X)[0]
    probs = emotion_clf.predict_proba(X)[0]
    return emotion, probs

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .header {
        font-size: 50px;
        color: #333333;
        text-align: center;
        padding: 20px;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 24px;
        color: #555555;
        text-align: center;
        margin-bottom: 20px;
    }
    .textbox-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #fff;
        padding: 10px;
        box-shadow: 0px -2px 5px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
    }
    .textbox {
        width: 85%;
        padding: 15px;
        border: 1px solid #ccc;
        border-radius: 10px;
        font-size: 16px;
        display: inline-block;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        border-radius: 5px;
        font-size: 16px;
        display: inline-block;
        vertical-align: top;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .chat-bubble {
        background-color: #e5e5ea;
        padding: 10px 15px;
        border-radius: 20px;
        margin-bottom: 10px;
        max-width: 60%;
    }
    .chat-bubble.user {
        background-color: #007BFF;
        color: white;
        margin-left: auto;
    }
    .chat-bubble.bot {
        background-color: #FFC107;
        color: black;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        padding-bottom: 150px;
    }
    .emotion-label {
        font-weight: bold;
        color: #FF4500;
    }
    .chat-label {
        font-weight: bold;
        margin-bottom: 5px;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown('<div class="header">Hi, I\'m HappyBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">I\'m Here To Make You Feel Happy!</div>', unsafe_allow_html=True)

# Initialize session state for storing chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# User input and submit button container
st.markdown('<div class="textbox-container">', unsafe_allow_html=True)
user_input = st.text_input('', '', key='textbox', placeholder='Type your message here...')
submit_button = st.button('Submit')
st.markdown('</div>', unsafe_allow_html=True)

if submit_button:
    if user_input:
        # Append user message to chat history
        st.session_state['chat_history'].append(('user', user_input))
        
        # Predict emotion and generate response
        emotion, probs = predict_emotion(user_input)
        dataset = pd.read_csv('dataset_with_emotions.csv')
        responses = dataset[dataset['Emotion'] == emotion]['Chatbot Response'].values
        response = random.choice(responses)
        
        # Append chatbot response and emotion to chat history
        st.session_state['chat_history'].append(('bot', response, emotion, probs))
    else:
        st.write('Please enter a message.')

# Layout for chat and pie chart
col1, col2 = st.columns([2, 1])

with col1:
    # Display chat history
    for message in st.session_state['chat_history']:
        if message[0] == 'user':
            st.markdown(f'<div class="chat-label">You:</div><div class="chat-bubble user">{message[1]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label">HappyBot:</div><div class="chat-bubble bot">{message[1]}<br><span class="emotion-label">Emotion: {message[2]}</span></div>', unsafe_allow_html=True)

# Display the most recent prediction pie chart in the bottom right corner
if len(st.session_state['chat_history']) > 0 and st.session_state['chat_history'][-1][0] == 'bot':
    latest_probs = st.session_state['chat_history'][-1][3]
    emotions = emotion_clf.classes_

    # Custom rainbow-like color sequence
    rainbow_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']

    # Plotting the prediction probabilities as a pie chart
    fig = px.pie(
        values=latest_probs,
        names=emotions,
        title='Emotion Prediction Probabilities',
        color_discrete_sequence=rainbow_colors
    )
    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        height=400,
        width=400,
        margin=dict(l=0, r=0, t=30, b=0),
        title_font_color='white',
        font=dict(color='white')
    )
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0, 0, 0, 0])

    st.markdown('<div class="element-container"><div class="element"></div></div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
