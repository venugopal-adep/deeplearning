import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, X_val, y_val, early_stopping, patience):
    model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)

    train_scores = []
    val_scores = []
    best_val_score = 0
    counter = 0

    for epoch in range(500):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))

        train_score = accuracy_score(y_train, model.predict(X_train))
        val_score = accuracy_score(y_val, model.predict(X_val))

        train_scores.append(train_score)
        val_scores.append(val_score)

        if early_stopping:
            if val_score > best_val_score:
                best_val_score = val_score
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    return model, train_scores, val_scores

def plot_learning_curve(train_scores, val_scores, early_stopping_point=None):
    epochs = list(range(1, len(train_scores) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_scores, mode='lines', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=epochs, y=val_scores, mode='lines', name='Validation Accuracy'))

    if early_stopping_point:
        fig.add_shape(type='line', x0=early_stopping_point, y0=0, x1=early_stopping_point, y1=1,
                      line=dict(color='red', width=2, dash='dash'), name='Early Stopping')

    fig.update_layout(title='Learning Curve', xaxis_title='Epoch', yaxis_title='Accuracy')
    return fig

# Generate dataset
X, y = make_classification(n_samples=1000, n_classes=2, n_features=10, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Streamlit app
st.write("## Early Stopping in Neural Networks")
st.write('**Developed by : Venugopal Adep**')

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
early_stopping = st.sidebar.checkbox("Enable Early Stopping", value=True)
patience = st.sidebar.slider("Patience", 1, 20, 5)

# Train the model
model, train_scores, val_scores = train_model(X_train, y_train, X_val, y_val, early_stopping, patience)

# Find the early stopping point
early_stopping_point = None
if early_stopping:
    early_stopping_point = len(val_scores) - patience

# Plot the learning curve
fig = plot_learning_curve(train_scores, val_scores, early_stopping_point)
st.plotly_chart(fig)

# Evaluate the model
train_accuracy = accuracy_score(y_train, model.predict(X_train))
val_accuracy = accuracy_score(y_val, model.predict(X_val))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

st.write(f"Training Accuracy: {train_accuracy:.4f}")
st.write(f"Validation Accuracy: {val_accuracy:.4f}")
st.write(f"Test Accuracy: {test_accuracy:.4f}")

# Explanations and examples
st.header("Early Stopping Explained")
st.write("Early stopping is a regularization technique used to prevent overfitting in neural networks. It involves monitoring the model's performance on a validation set during training and stopping the training process when the performance starts to degrade.")

st.subheader("How Early Stopping Works")
st.write("1. Split the data into training, validation, and test sets.")
st.write("2. Train the model on the training set and evaluate its performance on the validation set after each epoch.")
st.write("3. If the validation performance starts to worsen, keep track of the number of epochs without improvement.")
st.write("4. If the number of epochs without improvement reaches the specified patience, stop the training.")
st.write("5. Use the model with the best validation performance for final evaluation on the test set.")

st.subheader("Benefits of Early Stopping")
st.write("- Prevents overfitting by stopping the training process before the model starts to memorize the training data.")
st.write("- Helps to find the optimal number of training epochs automatically.")
st.write("- Reduces the training time by avoiding unnecessary epochs.")

st.subheader("Example")
st.write("In this example, we train a neural network on a binary classification task and demonstrate the effect of early stopping.")
st.write("- Enable the 'Early Stopping' checkbox to apply early stopping during training.")
st.write("- Adjust the 'Patience' slider to set the number of epochs to wait for improvement before stopping.")
st.write("The learning curve plot shows the training and validation accuracies over epochs. If early stopping is enabled, a vertical dashed line indicates the point at which training is stopped.")