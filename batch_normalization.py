import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

def normalize_batch(batch, gamma, beta):
    mean = np.mean(batch, axis=0)
    var = np.var(batch, axis=0)
    normalized_batch = (batch - mean) / np.sqrt(var + 1e-8)
    output = gamma * normalized_batch + beta
    return output

def plot_distributions(before, after):
    x_before = np.linspace(np.min(before), np.max(before), 100)
    x_after = np.linspace(np.min(after), np.max(after), 100)

    pdf_before = norm.pdf(x_before, loc=np.mean(before), scale=np.std(before))
    pdf_after = norm.pdf(x_after, loc=np.mean(after), scale=np.std(after))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_before, y=pdf_before, mode='lines', name='Before Normalization'))
    fig.add_trace(go.Scatter(x=x_after, y=pdf_after, mode='lines', name='After Normalization'))

    fig.add_shape(type='line', x0=np.mean(before), y0=0, x1=np.mean(before), y1=np.max(pdf_before),
                  line=dict(color='blue', width=2, dash='dash'), name='Mean (Before)')
    fig.add_shape(type='line', x0=np.mean(after), y0=0, x1=np.mean(after), y1=np.max(pdf_after),
                  line=dict(color='red', width=2, dash='dash'), name='Mean (After)')

    fig.update_layout(title='Probability Distributions',
                      xaxis_title='Activation',
                      yaxis_title='Probability Density',
                      showlegend=True)

    return fig

# Streamlit app
st.write("## Batch Normalization in Neural Networks")
st.write("**Developed by : Venugopal Adep**")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
num_samples = st.sidebar.slider("Number of samples", 1, 1000, 100)
num_neurons = st.sidebar.slider("Number of neurons", 1, 100, 10)
gamma = st.sidebar.slider("Gamma", 0.1, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("Beta", -2.0, 2.0, 0.0, 0.1)

# Generate random batch of activations with a wider range
hidden_before = np.random.normal(loc=2.0, scale=1.5, size=(num_samples, num_neurons))

# Apply batch normalization
hidden_after = normalize_batch(hidden_before, gamma, beta)

# Plot the probability distributions before and after normalization
fig = plot_distributions(hidden_before.flatten(), hidden_after.flatten())
st.plotly_chart(fig)

# Explanations and examples
st.header("Batch Normalization Explained")
st.write("Batch Normalization is a technique used in deep neural networks to normalize the inputs of each layer. It helps to stabilize and speed up the training process.")

st.subheader("Why Batch Normalization?")
st.write("During training, the distribution of inputs to each layer can shift as the parameters of the previous layers change. This phenomenon is known as internal covariate shift. Batch Normalization addresses this issue by normalizing the inputs to have zero mean and unit variance.")

st.subheader("How Batch Normalization Works")
st.write("1. For each batch of data, compute the mean and variance of the activations.")
st.write("2. Normalize the activations by subtracting the mean and dividing by the square root of the variance.")
st.write("3. Scale and shift the normalized activations using learnable parameters gamma and beta.")
st.write("4. Use the normalized activations as inputs to the next layer.")

st.subheader("Benefits of Batch Normalization")
st.write("- Reduces internal covariate shift, allowing for faster learning.")
st.write("- Enables the use of higher learning rates, leading to faster convergence.")
st.write("- Reduces the sensitivity to weight initialization.")
st.write("- Acts as a regularizer, reducing the need for techniques like dropout.")

st.subheader("Example")
st.write("Consider a batch of activations from a hidden layer:")
st.write(hidden_before[:5])
st.write("After applying Batch Normalization with gamma={} and beta={}:".format(gamma, beta))
st.write(hidden_after[:5])
st.write("Notice how the activations are normalized to have zero mean and unit variance, and then scaled and shifted by gamma and beta.")