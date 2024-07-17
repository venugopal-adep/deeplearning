import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_breast_cancer, load_iris
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Set page config
st.set_page_config(layout="wide", page_title="Neural Network Classification Explorer", page_icon="🧠")

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
st.markdown("<h1 style='text-align: center;'>🧠 Neural Network Classification Explorer 🧠</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Neural Network Classification Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the power of Neural Networks for classification tasks.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Neural Networks?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
Neural Networks are a set of algorithms inspired by the human brain, designed to recognize patterns. Key points:

- Composed of layers of interconnected nodes (neurons)
- Can learn complex non-linear relationships in data
- Require careful tuning of hyperparameters for optimal performance
- Capable of automatic feature extraction
- Widely used in various domains including computer vision, natural language processing, and more

Neural Networks have shown remarkable performance in many machine learning tasks, especially with large amounts of data.
</p>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Model Training", "🎛️ Hyperparameter Tuning", "📊 Model Visualization", "🧠 Quiz"])

# Load data
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Iris":
        data = load_iris()
    else:
        raise ValueError("Unknown dataset")
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

def create_model(input_dim, hidden_layers, neurons_per_layer, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

with tab1:
    st.markdown("<p class='medium-font'>Neural Network Model Training</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train a Neural Network model on a selected dataset and evaluate its performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"])
        X, y = load_data(dataset)
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        hidden_layers = st.slider("Number of hidden layers", 1, 5, 2)
        neurons_per_layer = st.slider("Neurons per hidden layer", 8, 128, 64, 8)
        dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.1)
        learning_rate = st.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f")
        epochs = st.slider("Number of epochs", 10, 200, 50, 10)
        
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = create_model(X_train.shape[1], hidden_layers, neurons_per_layer, dropout_rate, learning_rate)
            
            history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, verbose=0)
            
            train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            
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
            
            # Confusion Matrix
            y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'])
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Hyperparameter Tuning</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's explore how different hyperparameters affect the model's performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="hp_dataset")
        X, y = load_data(dataset)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        param_to_tune = st.selectbox("Parameter to tune", ["hidden_layers", "neurons_per_layer", "dropout_rate", "learning_rate"])
        epochs = st.slider("Number of epochs", 10, 200, 50, 10, key="hp_epochs")
        
        if st.button("Tune Hyperparameter"):
            results = []
            if param_to_tune == "hidden_layers":
                for hidden_layers in range(1, 6):
                    model = create_model(X_train.shape[1], hidden_layers, 64, 0.2, 0.001)
                    history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, verbose=0)
                    _, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                    results.append((hidden_layers, test_accuracy))
            elif param_to_tune == "neurons_per_layer":
                for neurons in range(8, 129, 8):
                    model = create_model(X_train.shape[1], 2, neurons, 0.2, 0.001)
                    history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, verbose=0)
                    _, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                    results.append((neurons, test_accuracy))
            elif param_to_tune == "dropout_rate":
                for dropout in np.arange(0, 0.6, 0.1):
                    model = create_model(X_train.shape[1], 2, 64, dropout, 0.001)
                    history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, verbose=0)
                    _, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                    results.append((dropout, test_accuracy))
            else:  # learning_rate
                for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
                    model = create_model(X_train.shape[1], 2, 64, 0.2, lr)
                    history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, verbose=0)
                    _, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                    results.append((lr, test_accuracy))
            
            results_df = pd.DataFrame(results, columns=[param_to_tune, 'Test Accuracy'])
            st.dataframe(results_df)

    with col2:
        if 'results_df' in locals():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df[param_to_tune], y=results_df['Test Accuracy'], mode='lines+markers'))
            fig.update_layout(
                title=f'Model Performance vs {param_to_tune}',
                xaxis_title=param_to_tune,
                yaxis_title='Test Accuracy'
            )
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Model Visualization</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's visualize the architecture of our Neural Network model.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="vis_dataset")
        X, y = load_data(dataset)
        
        hidden_layers = st.slider("Number of hidden layers", 1, 5, 2, key="vis_hidden_layers")
        neurons_per_layer = st.slider("Neurons per hidden layer", 8, 128, 64, 8, key="vis_neurons")
        
        if st.button("Visualize Model"):
            model = create_model(X.shape[1], hidden_layers, neurons_per_layer, 0.2, 0.001)
            
            # Create a string representation of the model architecture
            architecture = f"Input Layer ({X.shape[1]}) → "
            for i in range(hidden_layers):
                architecture += f"Dense ({neurons_per_layer}) → Dropout (0.2) → "
            architecture += "Output Layer (1)"
            
            st.markdown(f"""
            <p class='small-font'>
            Model Architecture:<br>
            {architecture}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'model' in locals():
            # Create a simple visualization of the model architecture
            fig = go.Figure()
            
            layer_sizes = [X.shape[1]] + [neurons_per_layer] * hidden_layers + [1]
            max_size = max(layer_sizes)
            
            for i, size in enumerate(layer_sizes):
                y = np.linspace(0, 1, size)
                x = [i] * size
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=10), name=f'Layer {i}'))
                
                if i < len(layer_sizes) - 1:
                    for j in range(size):
                        for k in range(layer_sizes[i+1]):
                            fig.add_trace(go.Scatter(x=[i, i+1], y=[y[j], np.linspace(0, 1, layer_sizes[i+1])[k]],
                                                     mode='lines', line=dict(width=0.5, color='gray'), showlegend=False))
            
            fig.update_layout(
                title='Neural Network Architecture',
                xaxis=dict(showticklabels=False, title='Layers'),
                yaxis=dict(showticklabels=False, title='Neurons'),
                showlegend=False,
                height=600
            )
            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is a neuron in a neural network?",
            "options": [
                "A biological cell in the brain",
                "A mathematical function that processes inputs and produces an output",
                "A type of activation function",
                "The final layer of the network"
            ],
            "correct": 1,
            "explanation": "In a neural network, a neuron is a mathematical function that takes in inputs, applies weights and biases, and produces an output, often through an activation function."
        },
        {
            "question": "What is the purpose of the activation function in a neural network?",
            "options": [
                "To initialize the weights of the network",
                "To introduce non-linearity into the network",
                "To calculate the loss of the model",
                "To normalize the input data"
            ],
            "correct": 1,
            "explanation": "The activation function introduces non-linearity into the network, allowing it to learn complex patterns and relationships in the data."
        },
        {
            "question": "What is the role of dropout in neural networks?",
            "options": [
                "To increase the learning rate",
                "To add more neurons to the network",
                "To prevent overfitting by randomly deactivating neurons during training",
                "To initialize the weights of the network"
            ],
            "correct": 2,
            "explanation": "Dropout is a regularization technique that helps prevent overfitting by randomly deactivating a portion of neurons during training, forcing the network to learn more robust features."
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
You've explored Neural Network classification through interactive examples and visualizations. 
Neural Networks are powerful tools for various machine learning tasks, especially when dealing with complex, 
high-dimensional data. Remember these key points:

1. Neural Networks can learn complex non-linear relationships in data.
2. The architecture (number of layers and neurons) can significantly impact performance.
3. Regularization techniques like dropout help prevent overfitting.
4. Hyperparameter tuning is crucial for optimal performance.
5. Visualization tools can help understand the model's architecture and training process.

Keep exploring and applying these concepts to solve real-world problems!
</p>
""", unsafe_allow_html=True)

# Add a footnote about TensorFlow
st.markdown("""
<p class='small-font' style='text-align: center; color: gray;'>
This app uses TensorFlow for neural network implementation. TensorFlow is an open-source machine learning library 
developed by Google Brain team for numerical computation and large-scale machine learning.
</p>
""", unsafe_allow_html=True)