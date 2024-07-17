import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff

# Set page config
st.set_page_config(layout="wide", page_title="Attention Mechanisms Explorer", page_icon="🔍")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #6A5ACD;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
        color: #483D8B;
    }
    .small-font {
        font-size: 16px !important;
    }
    .highlight {
        background-color: #E6E6FA;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #9370DB;
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
        background-color: #E6E6FA;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🔍 Attention Mechanisms Explorer 🔍</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Attention Mechanisms Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's dive into the fascinating world of Attention Mechanisms and see how they enhance neural networks for tasks like text summarization.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Attention Mechanisms?</p>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
<p class='small-font'>
Attention Mechanisms are a powerful technique in deep learning that help models focus on the most relevant parts of the input. Key points:

- They allow the model to dynamically focus on different parts of the input for each element of the output
- Particularly useful in tasks like machine translation, text summarization, and image captioning
- Improve the model's ability to handle long-range dependencies in sequences
- Can provide interpretability by showing which parts of the input the model is focusing on

Imagine Attention Mechanisms as a spotlight that can highlight the most important actors on a stage at different moments in a play.

In our example, we'll use Attention Mechanisms for text summarization, creating short summaries of longer texts.
</p>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    # Using a sample dataset of news articles and their summaries
    df = pd.read_csv("https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv", encoding='latin-1')
    df = df.dropna()
    df = df.head(1000)  # Using a subset for faster processing
    return df

df = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🏋️ Model Training & Prediction", "🧠 Quiz"])

with tab1:
    st.markdown("<p class='medium-font'>News Articles and Summaries Data Exploration</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        We're using a dataset of news articles and their summaries. Let's explore some characteristics of this dataset!
        </p>
        """, unsafe_allow_html=True)

        st.write(f"Dataset shape: {df.shape}")
        st.write(f"Number of unique articles: {df['text'].nunique()}")
        
    with col2:
        article_lengths = df['text'].str.split().str.len()
        summary_lengths = df['headlines'].str.split().str.len()
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=article_lengths, name='Article Lengths'))
        fig.add_trace(go.Box(y=summary_lengths, name='Summary Lengths'))
        fig.update_layout(
            title='Distribution of Article and Summary Lengths',
            yaxis_title='Number of Words'
        )
        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Attention Model Training & Prediction</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    # Initialize session state to store the model
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train an Attention-based model for text summarization.
        You can adjust some hyperparameters and see how they affect the model's performance.
        </p>
        """, unsafe_allow_html=True)

        max_words = st.slider("Maximum vocabulary size", 1000, 10000, 5000, 1000)
        embedding_dim = st.slider("Embedding dimension", 16, 128, 64, 16)
        lstm_units = st.slider("LSTM units", 16, 128, 64, 16)
        epochs = st.slider("Number of epochs", 1, 10, 3)
        
        if st.button("Train Model"):
            with st.spinner('Training model... This may take a moment.'):
                # Prepare data
                tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
                tokenizer.fit_on_texts(df['text'])
                
                X = tokenizer.texts_to_sequences(df['text'])
                X = pad_sequences(X, padding='post', maxlen=100)
                
                y = tokenizer.texts_to_sequences(df['headlines'])
                y = pad_sequences(y, padding='post', maxlen=15)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Build model
                inputs = Input(shape=(100,))
                embedding = Embedding(max_words, embedding_dim, input_length=100)(inputs)
                lstm = LSTM(lstm_units, return_sequences=True)(embedding)
                attention = Attention()([lstm, lstm])
                context_vector = Concatenate()([lstm, attention])
                output = Dense(max_words, activation='softmax')(context_vector)
                
                model = Model(inputs=inputs, outputs=output)
                
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
                
                # Store the model and tokenizer in session state
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.history = history
                
                st.success('Model training complete!')

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<p class='medium-font'>Text Summarization</p>", unsafe_allow_html=True)
        user_text = st.text_area("Enter a news article:", "The quick brown fox jumps over the lazy dog. This classic pangram contains every letter of the English alphabet at least once.")
        
        if st.button("Generate Summary"):
            if st.session_state.model is not None:
                # Tokenize and pad the input text
                input_sequence = st.session_state.tokenizer.texts_to_sequences([user_text])
                input_padded = pad_sequences(input_sequence, maxlen=100, padding='post')
                
                # Generate summary
                predicted_sequence = st.session_state.model.predict(input_padded)[0]
                predicted_words = [st.session_state.tokenizer.index_word[idx] for idx in predicted_sequence.argmax(axis=1) if idx != 0]
                summary = ' '.join(predicted_words)
                
                st.markdown(f"""
                <p class='small-font'>
                Generated Summary: {summary}
                </p>
                """, unsafe_allow_html=True)
                
                # Store the summary for visualization
                st.session_state.summary = summary
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

        if 'summary' in st.session_state:
            st.markdown("<p class='medium-font'>Attention Visualization</p>", unsafe_allow_html=True)
            # This is a placeholder for attention visualization
            # In a real application, you would extract attention weights from the model
            # and create a heatmap to show which words the model focused on
            words = user_text.split()
            attention_weights = np.random.rand(len(words))
            attention_weights = attention_weights / np.sum(attention_weights)
            
            fig = go.Figure(data=go.Heatmap(
                z=[attention_weights],
                x=words,
                colorscale='Viridis'))
            fig.update_layout(title='Attention Weights',
                              xaxis_title='Words',
                              yaxis_title='Attention')
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the main purpose of Attention Mechanisms in neural networks?",
            "options": [
                "To speed up training time",
                "To reduce the number of parameters",
                "To focus on the most relevant parts of the input",
                "To eliminate the need for data preprocessing"
            ],
            "correct": 2,
            "explanation": "Attention Mechanisms allow the model to dynamically focus on the most relevant parts of the input for each element of the output."
        },
        {
            "question": "In which tasks are Attention Mechanisms particularly useful?",
            "options": [
                "Image classification",
                "Machine translation and text summarization",
                "Audio processing",
                "Reinforcement learning"
            ],
            "correct": 1,
            "explanation": "Attention Mechanisms are especially useful in sequence-to-sequence tasks like machine translation and text summarization."
        },
        {
            "question": "How do Attention Mechanisms improve model interpretability?",
            "options": [
                "By reducing model complexity",
                "By visualizing which parts of the input the model focuses on",
                "By automatically generating explanations",
                "By eliminating the need for hidden layers"
            ],
            "correct": 1,
            "explanation": "Attention Mechanisms allow us to visualize which parts of the input the model is focusing on, providing insights into its decision-making process."
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
You've explored Attention Mechanisms and their application in text summarization. Remember these key points:

1. Attention Mechanisms allow models to focus on the most relevant parts of the input.
2. They're particularly useful in sequence-to-sequence tasks like machine translation and text summarization.
3. Attention can improve a model's ability to handle long-range dependencies in sequences.
4. Visualizing attention weights can provide insights into the model's decision-making process.
5. Hyperparameter tuning (e.g., embedding dimension, number of LSTM units) can significantly impact model performance.

Keep exploring and applying Attention Mechanisms to various natural language processing tasks!
</p>
</div>
""", unsafe_allow_html=True)

# Add a footnote about the libraries and dataset used
st.markdown("""
<p class='small-font' style='text-align: center; color: gray;'>
This app uses TensorFlow and Keras for implementing Attention Mechanisms, and a public dataset of news articles and summaries.
Plotly is used for interactive visualizations. These tools make it easier to explore and understand complex deep learning concepts.
</p>
""", unsafe_allow_html=True)