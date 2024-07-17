import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Set page config
st.set_page_config(layout="wide", page_title="Neural Networks Basics Explorer", page_icon="🧠")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .small-font {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🧠 Neural Networks Basics Explorer 🧠</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Neural Networks Basics Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the fundamentals of Neural Networks using the MNIST digits dataset.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Neural Networks?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
Neural Networks are a type of machine learning algorithm inspired by the human brain. They consist of:

1. Input Layer: Receives the initial data
2. Hidden Layers: Process the data through weighted connections
3. Output Layer: Produces the final prediction
4. Activation Functions: Introduce non-linearity into the model
5. Weights and Biases: Adjusted during training to improve predictions

Neural Networks can learn complex patterns in data, making them suitable for various tasks like image recognition, natural language processing, and more.
</p>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Data Exploration", "🏋️ Model Training", "📊 Results Visualization", "🧠 Quiz"])

# Load data
@st.cache_data
def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y

X, y = load_data()

def create_model(input_dim, num_classes, hidden_layers, neurons_per_layer, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

with tab1:
    st.markdown("<p class='medium-font'>Data Exploration</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's explore the MNIST digits dataset, which contains images of handwritten digits (0-9).
        </p>
        """, unsafe_allow_html=True)

        st.write(f"Dataset shape: {X.shape}")
        st.write(f"Number of classes: {len(np.unique(y))}")
        
        sample_index = st.slider("Select a sample image", 0, len(X)-1, 42)
        
    with col2:
        fig = go.Figure(data=go.Heatmap(
            z=X[sample_index].reshape(8, 8),
            colorscale='Greys',
        ))
        fig.update_layout(
            title=f'Sample Digit: {y[sample_index]}',
            xaxis_title="Pixel X",
            yaxis_title="Pixel Y"
        )
        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Model Training</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train a Neural Network model on the MNIST digits dataset.
        </p>
        """, unsafe_allow_html=True)

        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        hidden_layers = st.slider("Number of hidden layers", 1, 5, 2)
        neurons_per_layer = st.slider("Neurons per hidden layer", 16, 256, 64, 16)
        dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.1)
        learning_rate = st.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f")
        epochs = st.slider("Number of epochs", 10, 100, 50, 10)
        
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            y_train_cat = to_categorical(y_train)
            y_test_cat = to_categorical(y_test)
            
            model = create_model(X_train.shape[1], 10, hidden_layers, neurons_per_layer, dropout_rate, learning_rate)
            
            history = model.fit(X_train_scaled, y_train_cat, epochs=epochs, validation_split=0.2, verbose=0)
            
            train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train_cat, verbose=0)
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
            
            st.markdown(f"""
            <p class='small-font'>
            Train Accuracy: {train_accuracy:.4f}<br>
            Test Accuracy: {test_accuracy:.4f}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'history' in locals():
            # Training history plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Accuracy'))
            fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'))
            fig.update_layout(title='Model Accuracy over Epochs',
                              xaxis_title='Epoch',
                              yaxis_title='Accuracy')
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Results Visualization</p>", unsafe_allow_html=True)
    
    if 'model' in locals():
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <p class='small-font'>
            Let's visualize the model's predictions and performance.
            </p>
            """, unsafe_allow_html=True)

            y_pred = model.predict(X_test_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            cm = confusion_matrix(y_test, y_pred_classes)
            fig = ff.create_annotated_heatmap(cm, x=list(range(10)), y=list(range(10)))
            fig.update_layout(title='Confusion Matrix',
                              xaxis_title='Predicted',
                              yaxis_title='Actual')
            st.plotly_chart(fig)

        with col2:
            sample_index = st.slider("Select a test sample", 0, len(X_test)-1, 0, key="test_sample")
            
            fig = go.Figure(data=go.Heatmap(
                z=X_test[sample_index].reshape(8, 8),
                colorscale='Greys',
            ))
            fig.update_layout(
                title=f'Test Sample (Actual: {y_test[sample_index]}, Predicted: {y_pred_classes[sample_index]})',
                xaxis_title="Pixel X",
                yaxis_title="Pixel Y"
            )
            st.plotly_chart(fig)

            st.markdown(f"""
            <p class='small-font'>
            Prediction Probabilities:
            </p>
            """, unsafe_allow_html=True)
            
            fig = go.Figure(data=go.Bar(x=list(range(10)), y=y_pred[sample_index]))
            fig.update_layout(title='Prediction Probabilities',
                              xaxis_title='Digit',
                              yaxis_title='Probability')
            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the purpose of the activation function in a neural network?",
            "options": [
                "To initialize the weights",
                "To introduce non-linearity",
                "To normalize the input data",
                "To calculate the loss"
            ],
            "correct": 1,
            "explanation": "The activation function introduces non-linearity into the network, allowing it to learn complex patterns in the data."
        },
        {
            "question": "What does the dropout layer do in a neural network?",
            "options": [
                "Adds more neurons to the network",
                "Removes neurons from the network permanently",
                "Randomly deactivates neurons during training to prevent overfitting",
                "Increases the learning rate"
            ],
            "correct": 2,
            "explanation": "Dropout randomly deactivates a portion of neurons during training, which helps prevent overfitting by forcing the network to learn more robust features."
        },
        {
            "question": "In the context of neural networks, what does 'epoch' mean?",
            "options": [
                "The number of layers in the network",
                "The learning rate of the model",
                "One complete pass through the entire training dataset",
                "The accuracy of the model"
            ],
            "correct": 2,
            "explanation": "An epoch represents one complete pass through the entire training dataset during the training process."
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
<p class='small-font'>
You've explored the basics of Neural Networks through interactive examples and visualizations. 
Remember these key points:

1. Neural Networks consist of interconnected layers of neurons.
2. They can learn complex patterns in data through training.
3. Activation functions introduce non-linearity, allowing the network to learn complex relationships.
4. Dropout is a technique to prevent overfitting.
5. The architecture and hyperparameters of a neural network can significantly impact its performance.

Keep exploring and applying these concepts to solve various machine learning problems!
</p>
""", unsafe_allow_html=True)

# Add a footnote about TensorFlow and scikit-learn
st.markdown("""
<p class='small-font' style='text-align: center; color: gray;'>
This app uses TensorFlow for neural network implementation and scikit-learn for data handling and preprocessing. 
TensorFlow is an open-source machine learning library developed by Google Brain team, while scikit-learn is a 
machine learning library for Python.
</p>
""", unsafe_allow_html=True)