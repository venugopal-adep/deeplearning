import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.metrics import roc_curve, auc

# Set page config
st.set_page_config(layout="wide", page_title="Anomaly Detection with Autoencoders Explorer", page_icon="🕵️")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #4B0082;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
        color: #8A2BE2;
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
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🕵️ Anomaly Detection with Autoencoders Explorer 🕵️</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Anomaly Detection with Autoencoders Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's dive into the fascinating world of Anomaly Detection using Autoencoders and see how they can identify unusual patterns in data.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What is Anomaly Detection with Autoencoders?</p>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
<p class='small-font'>
Anomaly Detection with Autoencoders is a technique used to identify unusual patterns in data. Key points:

- Autoencoders are trained to reconstruct normal data
- Anomalies are detected when the reconstruction error is high
- Useful for fraud detection, system health monitoring, and outlier detection
- Can work with high-dimensional data and capture complex patterns

Imagine Anomaly Detection with Autoencoders as a smart security system that learns what's normal and flags anything unusual.

In our example, we'll use Autoencoders to detect fraudulent credit card transactions.
</p>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
    features = df.drop(['Class', 'Time'], axis=1)
    labels = df['Class']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, features.columns

X_train, X_test, y_train, y_test, feature_names = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🏋️ Model Training & Prediction", "🧠 Quiz"])

with tab1:
    st.markdown("<p class='medium-font'>Credit Card Fraud Data Exploration</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        We're using the Credit Card Fraud Detection dataset. Let's explore some characteristics of this dataset!
        </p>
        """, unsafe_allow_html=True)

        st.write(f"Dataset shape: {X_train.shape[0] + X_test.shape[0]} rows, {X_train.shape[1]} features")
        st.write(f"Number of fraudulent transactions: {sum(y_train) + sum(y_test)}")
        st.write(f"Percentage of fraudulent transactions: {((sum(y_train) + sum(y_test)) / (len(y_train) + len(y_test)) * 100):.2f}%")
        
    with col2:
        fig = px.histogram(pd.DataFrame(X_train[:, :2], columns=feature_names[:2]))
        fig.update_layout(title='Distribution of First Two Features')
        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Autoencoder Model Training & Anomaly Detection</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    # Initialize session state to store the model
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train an Autoencoder model to detect fraudulent credit card transactions.
        You can adjust some hyperparameters and see how they affect the model's performance.
        </p>
        """, unsafe_allow_html=True)

        encoding_dim = st.slider("Encoding dimension", 8, 24, 16, 2)
        epochs = st.slider("Number of epochs", 10, 100, 50, 10)
        threshold_percentile = st.slider("Anomaly threshold percentile", 90, 99, 95, 1)
        
        if st.button("Train Model"):
            with st.spinner('Training model... This may take a moment.'):
                # Build model
                input_dim = X_train.shape[1]
                input_layer = Input(shape=(input_dim,))
                encoded = Dense(encoding_dim, activation='relu')(input_layer)
                decoded = Dense(input_dim, activation='linear')(encoded)
                
                autoencoder = Model(input_layer, decoded)
                autoencoder.compile(optimizer='adam', loss='mean_squared_error')
                
                history = autoencoder.fit(X_train, X_train,
                                          epochs=epochs,
                                          batch_size=32,
                                          shuffle=True,
                                          validation_data=(X_test, X_test),
                                          verbose=0)
                
                # Compute reconstruction error
                reconstructed = autoencoder.predict(X_test)
                mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)
                threshold = np.percentile(mse, threshold_percentile)
                
                # Store the model and results in session state
                st.session_state.model = autoencoder
                st.session_state.history = history
                st.session_state.mse = mse
                st.session_state.threshold = threshold
                st.session_state.y_pred = (mse > threshold).astype(int)
                
                st.success('Model training complete!')

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<p class='medium-font'>Anomaly Detection Results</p>", unsafe_allow_html=True)
        
        if 'y_pred' in st.session_state:
            tp = np.sum((y_test == 1) & (st.session_state.y_pred == 1))
            fp = np.sum((y_test == 0) & (st.session_state.y_pred == 1))
            tn = np.sum((y_test == 0) & (st.session_state.y_pred == 0))
            fn = np.sum((y_test == 1) & (st.session_state.y_pred == 0))
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1-score: {f1_score:.4f}")

    with col2:
        if 'history' in st.session_state:
            st.markdown("<p class='medium-font'>Training Results</p>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.history.history['loss'], name='Train Loss'))
            fig.add_trace(go.Scatter(y=st.session_state.history.history['val_loss'], name='Validation Loss'))
            fig.update_layout(title='Model Loss over Epochs',
                              xaxis_title='Epoch',
                              yaxis_title='Loss')
            st.plotly_chart(fig)

        if 'mse' in st.session_state:
            st.markdown("<p class='medium-font'>Reconstruction Error Distribution</p>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=st.session_state.mse[y_test == 0], name='Normal', opacity=0.7))
            fig.add_trace(go.Histogram(x=st.session_state.mse[y_test == 1], name='Fraudulent', opacity=0.7))
            fig.add_trace(go.Scatter(x=[st.session_state.threshold, st.session_state.threshold], 
                                     y=[0, 100], mode='lines', name='Threshold'))
            fig.update_layout(title='Reconstruction Error Distribution',
                              xaxis_title='Reconstruction Error',
                              yaxis_title='Count',
                              barmode='overlay')
            st.plotly_chart(fig)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, st.session_state.mse)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
            fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the main principle behind using Autoencoders for Anomaly Detection?",
            "options": [
                "Autoencoders classify data into normal and anomalous categories",
                "Autoencoders compress data to reduce its dimensionality",
                "Autoencoders learn to reconstruct normal data, and anomalies have high reconstruction error",
                "Autoencoders generate new data samples to compare with existing ones"
            ],
            "correct": 2,
            "explanation": "Autoencoders are trained to reconstruct normal data. Anomalies are detected when the reconstruction error is high, indicating that the autoencoder struggles to reconstruct the input."
        },
        {
            "question": "What is the advantage of using Autoencoders for Anomaly Detection compared to traditional statistical methods?",
            "options": [
                "Autoencoders are always faster to train",
                "Autoencoders can capture complex, non-linear patterns in the data",
                "Autoencoders require less data to train",
                "Autoencoders always provide perfect anomaly detection"
            ],
            "correct": 1,
            "explanation": "Autoencoders can capture complex, non-linear patterns in the data, making them more flexible than many traditional statistical methods for anomaly detection."
        },
        {
            "question": "How is the anomaly threshold typically determined in Autoencoder-based Anomaly Detection?",
            "options": [
                "It's always set to 0.5",
                "By using a percentile of the reconstruction errors on normal data",
                "By randomly selecting a value",
                "It's always set to the mean of the reconstruction errors"
            ],
            "correct": 1,
            "explanation": "The anomaly threshold is often set using a high percentile (e.g., 95th or 99th) of the reconstruction errors on normal data. This allows for some flexibility in capturing rare but normal variations in the data."
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
You've explored Anomaly Detection with Autoencoders and their application in fraud detection. Remember these key points:

1. Autoencoders learn to reconstruct normal patterns in the data.
2. Anomalies are detected when the reconstruction error is high.
3. The method can capture complex, non-linear patterns in high-dimensional data.
4. The choice of encoding dimension and anomaly threshold can significantly impact performance.
5. Evaluation metrics like precision, recall, and ROC curves are crucial for assessing the model's effectiveness.

Keep exploring and applying Anomaly Detection with Autoencoders to various domains like cybersecurity, industrial monitoring, and medical diagnosis!
</p>
</div>
""", unsafe_allow_html=True)

# Add a footnote about the libraries and dataset used
st.markdown("""
<p class='small-font' style='text-align: center; color: gray;'>
This app uses TensorFlow and Keras for implementing Autoencoders, and the Credit Card Fraud Detection dataset from Kaggle.
Plotly is used for interactive visualizations. These tools make it easier to explore and understand complex machine learning concepts.
</p>
""", unsafe_allow_html=True)