import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense

# Load and preprocess the dataset
def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Build and train the autoencoder model
def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Streamlit app
def main():
    st.title("Autoencoder for Dimensionality Reduction")
    st.write("This demo shows how an Autoencoder can be used for dimensionality reduction.")

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data()

    # User input for encoding dimension
    encoding_dim = st.slider("Select the encoding dimension", min_value=2, max_value=64, value=32, step=1)

    # Build and train the autoencoder
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

    # Encode the training data
    encoder = Model(autoencoder.input, autoencoder.layers[1].output)
    encoded_data = encoder.predict(X_train)

    # Visualize the encoded data
    fig = px.scatter(x=encoded_data[:, 0], y=encoded_data[:, 1], color=y_train, labels={'color': 'Digit'})
    fig.update_layout(title=f"Encoded Data (Encoding Dimension: {encoding_dim})")
    st.plotly_chart(fig)

    # Reconstruct the test data
    reconstructed_data = autoencoder.predict(X_test)

    # Visualize the original and reconstructed images
    st.subheader("Original vs Reconstructed Images")
    for i in range(5):
        st.image([X_test[i].reshape(8, 8), reconstructed_data[i].reshape(8, 8)], caption=[f"Original Image {i+1}", f"Reconstructed Image {i+1}"], width=200)

if __name__ == '__main__':
    main()