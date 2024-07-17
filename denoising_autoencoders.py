import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.datasets import mnist
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="Denoising Autoencoders Explorer", page_icon="🖼️")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #FF6347;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
        color: #FF7F50;
    }
    .small-font {
        font-size: 16px !important;
    }
    .highlight {
        background-color: #FFF5EE;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #FFA07A;
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
        background-color: #FFF5EE;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #FF4500;'>🖼️ Denoising Autoencoders Explorer 🖼️</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Denoising Autoencoders Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's dive into the fascinating world of Denoising Autoencoders and see how they can clean up noisy images.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Denoising Autoencoders?</p>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
<p class='small-font'>
Denoising Autoencoders are a type of neural network designed to remove noise from data. Key points:

- They learn to reconstruct clean data from noisy input
- Useful for image denoising, data cleaning, and feature learning
- Consist of an encoder (compresses the input) and a decoder (reconstructs the output)
- Can learn robust features that are resistant to noise

Imagine Denoising Autoencoders as a smart photo restoration tool that can clean up old, grainy photographs.

In our example, we'll use Denoising Autoencoders to remove noise from handwritten digit images.
</p>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return x_train, x_test

x_train, x_test = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🏋️ Model Training & Prediction", "🧠 Quiz"])

with tab1:
    st.markdown("<p class='medium-font'>MNIST Handwritten Digits Data Exploration</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        We're using the MNIST dataset of handwritten digits. Let's explore some characteristics of this dataset!
        </p>
        """, unsafe_allow_html=True)

        st.write(f"Dataset shape: {x_train.shape}")
        st.write(f"Number of training images: {len(x_train)}")
        st.write(f"Number of test images: {len(x_test)}")
        
    with col2:
        fig = px.imshow(x_train[0].reshape(28, 28), color_continuous_scale='gray')
        fig.update_layout(title='Sample MNIST Digit', coloraxis_showscale=False)
        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Denoising Autoencoder Model Training & Prediction</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    # Initialize session state to store the model
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train a Denoising Autoencoder model to remove noise from MNIST digits.
        You can adjust some hyperparameters and see how they affect the model's performance.
        </p>
        """, unsafe_allow_html=True)

        noise_factor = st.slider("Noise factor", 0.1, 0.5, 0.3, 0.1)
        latent_dim = st.slider("Latent dimension", 8, 64, 32, 8)
        epochs = st.slider("Number of epochs", 1, 10, 3)
        
        if st.button("Train Model"):
            with st.spinner('Training model... This may take a moment.'):
                # Add noise to the data
                x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
                x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
                x_train_noisy = np.clip(x_train_noisy, 0., 1.)
                x_test_noisy = np.clip(x_test_noisy, 0., 1.)
                
                # Build model
                input_img = Input(shape=(28, 28, 1))
                x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
                x = MaxPooling2D((2, 2), padding='same')(x)
                x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                encoded = MaxPooling2D((2, 2), padding='same')(x)

                x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
                x = UpSampling2D((2, 2))(x)
                x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = UpSampling2D((2, 2))(x)
                decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

                autoencoder = Model(input_img, decoded)
                autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
                
                history = autoencoder.fit(x_train_noisy, x_train,
                                          epochs=epochs,
                                          batch_size=128,
                                          shuffle=True,
                                          validation_data=(x_test_noisy, x_test),
                                          verbose=0)
                
                # Store the model in session state
                st.session_state.model = autoencoder
                st.session_state.history = history
                st.session_state.x_test_noisy = x_test_noisy
                
                st.success('Model training complete!')

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<p class='medium-font'>Image Denoising</p>", unsafe_allow_html=True)
        
        if st.button("Denoise Random Image"):
            if st.session_state.model is not None:
                # Select a random noisy image
                idx = np.random.randint(0, len(x_test))
                noisy_image = st.session_state.x_test_noisy[idx]
                
                # Denoise the image
                denoised_image = st.session_state.model.predict(noisy_image.reshape(1, 28, 28, 1))[0]
                
                # Store the images for visualization
                st.session_state.noisy_image = noisy_image
                st.session_state.denoised_image = denoised_image
                st.session_state.original_image = x_test[idx]
            else:
                st.warning("Please train a model first.")

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

        if 'noisy_image' in st.session_state:
            st.markdown("<p class='medium-font'>Denoising Results</p>", unsafe_allow_html=True)
            fig = px.imshow(np.hstack((st.session_state.original_image.reshape(28, 28),
                                       st.session_state.noisy_image.reshape(28, 28),
                                       st.session_state.denoised_image.reshape(28, 28))),
                            color_continuous_scale='gray')
            fig.update_layout(title='Original, Noisy, and Denoised Images')
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the main purpose of Denoising Autoencoders?",
            "options": [
                "To classify images",
                "To generate new images",
                "To remove noise from data",
                "To compress data"
            ],
            "correct": 2,
            "explanation": "Denoising Autoencoders are primarily designed to reconstruct clean data from noisy input, effectively removing noise from the data."
        },
        {
            "question": "What are the two main components of an autoencoder?",
            "options": [
                "Classifier and Regressor",
                "Encoder and Decoder",
                "Convolution and Pooling",
                "Input and Output"
            ],
            "correct": 1,
            "explanation": "Autoencoders consist of an encoder that compresses the input data and a decoder that reconstructs the output from the compressed representation."
        },
        {
            "question": "How can Denoising Autoencoders be useful beyond just denoising?",
            "options": [
                "They can be used for feature learning",
                "They can generate music",
                "They can predict stock prices",
                "They can translate languages"
            ],
            "correct": 0,
            "explanation": "Denoising Autoencoders can learn robust features that are resistant to noise, making them useful for feature learning in various applications."
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
You've explored Denoising Autoencoders and their application in image denoising. Remember these key points:

1. Denoising Autoencoders learn to reconstruct clean data from noisy input.
2. They consist of an encoder that compresses the input and a decoder that reconstructs the output.
3. They're useful for image denoising, data cleaning, and feature learning.
4. The noise factor and network architecture can significantly impact the model's performance.
5. Denoising Autoencoders can learn robust features that are resistant to noise.

Keep exploring and applying Denoising Autoencoders to various data cleaning and feature learning tasks!
</p>
</div>
""", unsafe_allow_html=True)

# Add a footnote about the libraries and dataset used
st.markdown("""
<p class='small-font' style='text-align: center; color: gray;'>
This app uses TensorFlow and Keras for implementing Denoising Autoencoders, and the MNIST dataset of handwritten digits.
Plotly is used for interactive visualizations. These tools make it easier to explore and understand complex deep learning concepts.
</p>
""", unsafe_allow_html=True)