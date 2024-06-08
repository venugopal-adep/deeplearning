import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def apply_dropout(neurons, dropout_rate):
    num_dropped = int(len(neurons) * dropout_rate)
    dropped_indices = np.random.choice(len(neurons), num_dropped, replace=False)
    neurons[dropped_indices] = 0
    return neurons

def plot_decision_boundary(model, X, y, dropout_rate=0.0):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig = go.Figure(data=go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=True))
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, size=10, line=dict(width=1, color='white'))))
    fig.update_layout(title=f'Decision Boundary (Dropout Rate: {dropout_rate})', xaxis_title='X1', yaxis_title='X2')
    return fig

# Generate dataset
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
st.write("## Dropout in Neural Networks")
st.write("**Developed by : Venugopal Adep**")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 1.0, 0.0, 0.1)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Apply dropout to the trained model
for layer in model.coefs_:
    layer *= np.random.binomial(1, 1 - dropout_rate, size=layer.shape) / (1 - dropout_rate)

# Make predictions and calculate accuracy
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Plot the decision boundary
fig = plot_decision_boundary(model, X, y, dropout_rate)
st.plotly_chart(fig)

# Display the accuracy scores
st.write(f"Training Accuracy: {train_accuracy:.2f}")
st.write(f"Test Accuracy: {test_accuracy:.2f}")

# Explanations and examples
st.header("Dropout Explained")
st.write("Dropout is a regularization technique used in neural networks to prevent overfitting. It randomly sets a fraction of the neurons to zero during each training step, which helps to reduce co-dependency among neurons and improves generalization.")

st.subheader("How Dropout Works")
st.write("1. During each training step, randomly select a fraction (dropout rate) of neurons to be dropped out.")
st.write("2. Set the values of the dropped-out neurons to zero, effectively deactivating them.")
st.write("3. Train the network with the remaining active neurons.")
st.write("4. During inference (prediction), multiply the neuron values by (1 - dropout rate) to compensate for the scaling.")

st.subheader("Benefits of Dropout")
st.write("- Reduces overfitting by preventing neurons from relying too much on each other.")
st.write("- Forces the network to learn more robust features that are useful in conjunction with different random subsets of neurons.")
st.write("- Provides a way of approximately combining many different neural network architectures efficiently.")

st.subheader("Example")
st.write("In this example, we train a neural network on a binary classification task (moon dataset) and visualize the effect of dropout on the decision boundary.")
st.write("Adjust the 'Dropout Rate' slider to see how dropout affects the network's performance and decision boundary.")