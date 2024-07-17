import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff

# Set page config
st.set_page_config(layout="wide", page_title="Recurrent Neural Networks Explorer", page_icon="🔁")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #1E90FF;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
        color: #4682B4;
    }
    .small-font {
        font-size: 16px !important;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    .stTextInput>div>div>input {
        background-color: #F0F8FF;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4169E1;'>🔁 Recurrent Neural Networks Explorer 🔁</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Recurrent Neural Networks Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the power of RNNs for sequence data processing, focusing on sentiment analysis of movie reviews.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Recurrent Neural Networks?</p>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
<p class='small-font'>
Recurrent Neural Networks (RNNs) are a class of neural networks designed to work with sequence data. Key points:

- They can process input of any length
- They maintain an internal state (memory) that gets updated as they process each element in the sequence
- Well-suited for tasks like natural language processing, time series analysis, and speech recognition
- Can suffer from vanishing/exploding gradient problems for very long sequences (addressed by variants like LSTM and GRU)

Imagine RNNs as a reader going through a book, remembering important details and using that information to understand the context of what they're currently reading.

In our example, we'll use RNNs for sentiment analysis of movie reviews, classifying them as positive or negative.
</p>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data(num_words=10000, maxlen=200):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X = pad_sequences(X, maxlen=maxlen)
    return X, y

X, y = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🏋️ Model Training & Prediction", "🧠 Quiz"])

with tab1:
    st.markdown("<p class='medium-font'>IMDB Movie Review Data Exploration</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        We're using the IMDB movie review dataset. It contains 50,000 movie reviews labeled as positive or negative.
        Let's explore some characteristics of this dataset!
        </p>
        """, unsafe_allow_html=True)

        st.write(f"Dataset shape: {X.shape}")
        st.write(f"Number of positive reviews: {sum(y == 1)}")
        st.write(f"Number of negative reviews: {sum(y == 0)}")
        
    with col2:
        review_lengths = [len(review) for review in X]
        fig = go.Figure(data=[go.Histogram(x=review_lengths)])
        fig.update_layout(
            title='Distribution of Review Lengths',
            xaxis_title='Review Length',
            yaxis_title='Count'
        )
        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>RNN Model Training & Prediction</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    # Initialize session state to store the model
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train an RNN model on the IMDB dataset for sentiment analysis.
        You can adjust some hyperparameters and see how they affect the model's performance.
        </p>
        """, unsafe_allow_html=True)

        embedding_dim = st.slider("Embedding dimension", 16, 128, 32, 16)
        rnn_units = st.slider("RNN units", 16, 128, 64, 16)
        learning_rate = st.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f")
        epochs = st.slider("Number of epochs", 1, 10, 3)
        
        if st.button("Train Model"):
            with st.spinner('Training model... This may take a moment.'):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = Sequential([
                    Embedding(10000, embedding_dim, input_length=200),
                    SimpleRNN(rnn_units),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer=Adam(learning_rate=learning_rate),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
                
                history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
                
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                
                # Store the model in session state
                st.session_state.model = model
                st.session_state.history = history
                
                st.success('Model training complete!')
                st.markdown(f"""
                <p class='small-font'>
                Test Accuracy: {test_accuracy:.4f}
                </p>
                """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<p class='medium-font'>Sentiment Prediction</p>", unsafe_allow_html=True)
        user_review = st.text_area("Enter a movie review:", "This movie was great! I really enjoyed it.")
        
        if st.button("Predict Sentiment"):
            if st.session_state.model is not None:
                # Convert the review to a sequence of word indices
                word_index = imdb.get_word_index()
                review_sequence = [word_index.get(word, 0) for word in user_review.lower().split()]
                review_sequence = pad_sequences([review_sequence], maxlen=200)
                
                prediction = st.session_state.model.predict(review_sequence)[0][0]
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                
                st.markdown(f"""
                <p class='small-font'>
                Predicted sentiment: {sentiment}<br>
                Confidence: {abs(prediction - 0.5) * 2:.2f}
                </p>
                """, unsafe_allow_html=True)
                
                # Store the prediction for visualization
                st.session_state.prediction = prediction
            else:
                st.warning("Please train a model first.")

    with col2:
        if 'history' in st.session_state:
            st.markdown("<p class='medium-font'>Training Results</p>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.history.history['accuracy'], name='Train Accuracy'))
            fig.add_trace(go.Scatter(y=st.session_state.history.history['val_accuracy'], name='Validation Accuracy'))
            fig.update_layout(title='Model Accuracy over Epochs',
                              xaxis_title='Epoch',
                              yaxis_title='Accuracy')
            st.plotly_chart(fig)

        if 'prediction' in st.session_state:
            st.markdown("<p class='medium-font'>Prediction Results</p>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=['Negative', 'Positive'], y=[1-st.session_state.prediction, st.session_state.prediction]))
            fig.update_layout(title='Sentiment Prediction',
                              xaxis_title='Sentiment',
                              yaxis_title='Probability')
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the main advantage of RNNs over traditional feedforward neural networks?",
            "options": [
                "They can process input of any length",
                "They always provide better accuracy",
                "They require less computational power",
                "They don't need training data"
            ],
            "correct": 0,
            "explanation": "RNNs can handle input sequences of varying lengths, making them suitable for tasks like natural language processing."
        },
        {
            "question": "What is the purpose of the embedding layer in our RNN model?",
            "options": [
                "To reduce the dimensionality of the input",
                "To convert words into dense vector representations",
                "To increase the model's accuracy",
                "To speed up the training process"
            ],
            "correct": 1,
            "explanation": "The embedding layer converts words (represented as indices) into dense vector representations, capturing semantic relationships between words."
        },
        {
            "question": "What is a potential drawback of basic RNNs when dealing with long sequences?",
            "options": [
                "They require too much memory",
                "They can't handle sequences longer than 1000 elements",
                "They may suffer from vanishing or exploding gradients",
                "They always overfit the training data"
            ],
            "correct": 2,
            "explanation": "Basic RNNs can struggle with long-term dependencies due to the vanishing or exploding gradient problem, which is addressed by variants like LSTM and GRU."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<p class='small-font'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.info(q['explanation'])
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='big-font'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()

# Conclusion
st.markdown("<p class='big-font'>Congratulations! 🎊</p>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
<p class='small-font'>
You've explored Recurrent Neural Networks and their application in sentiment analysis. Remember these key points:

1. RNNs are designed to work with sequence data, maintaining an internal state as they process each element.
2. They're well-suited for tasks like natural language processing, time series analysis, and speech recognition.
3. The embedding layer helps convert words into dense vector representations, capturing semantic relationships.
4. RNNs can struggle with long-term dependencies, which is addressed by variants like LSTM and GRU.
5. Hyperparameter tuning (e.g., embedding dimension, number of RNN units) can significantly impact model performance.

Keep exploring and applying RNNs to various sequence processing tasks!
</p>
</div>
""", unsafe_allow_html=True)

# Add a footnote about the libraries and dataset used
st.markdown("""
<p class='small-font' style='text-align: center; color: gray;'>
This app uses TensorFlow and Keras for implementing RNNs, and the IMDB movie review dataset for sentiment analysis.
Plotly is used for interactive visualizations. These tools make it easier to explore and understand complex deep learning concepts.
</p>
""", unsafe_allow_html=True)