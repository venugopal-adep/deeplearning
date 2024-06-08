import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_propagation(X, hidden_weights, hidden_bias, output_weights, output_bias, activation):
    hidden_layer = np.dot(X, hidden_weights) + hidden_bias
    hidden_activation = activation(hidden_layer)
    output_layer = np.dot(hidden_activation, output_weights) + output_bias
    if activation == softmax:
        output_activation = softmax(output_layer)
    else:
        output_activation = sigmoid(output_layer)
    return hidden_activation, output_activation

def plot_activations(layer_activations, layer_name):
    fig = go.Figure(data=[go.Bar(x=list(range(len(layer_activations[0]))), y=layer_activations[0])])
    fig.update_layout(title=f"{layer_name} Activations", xaxis_title="Neuron", yaxis_title="Activation")
    return fig

# Streamlit app
st.title("Forward Propagation in Neural Networks")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
num_features = st.sidebar.slider("Number of Input Features", 1, 10, 5)
num_hidden_neurons = st.sidebar.slider("Number of Hidden Neurons", 1, 10, 5)
num_output_neurons = st.sidebar.slider("Number of Output Neurons", 1, 10, 2)
activation_func = st.sidebar.selectbox("Activation Function", ("Sigmoid", "Softmax"))

# Generate random input data and weights
X = np.random.randn(1, num_features)
hidden_weights = np.random.randn(num_features, num_hidden_neurons)
hidden_bias = np.random.randn(1, num_hidden_neurons)
output_weights = np.random.randn(num_hidden_neurons, num_output_neurons)
output_bias = np.random.randn(1, num_output_neurons)

# Perform forward propagation
if activation_func == "Sigmoid":
    hidden_activation, output_activation = forward_propagation(X, hidden_weights, hidden_bias, output_weights, output_bias, sigmoid)
else:
    hidden_activation, output_activation = forward_propagation(X, hidden_weights, hidden_bias, output_weights, output_bias, softmax)

# Display the input features
st.subheader("Input Features")
st.write(X)

# Plot the hidden layer activations
fig_hidden = plot_activations(hidden_activation, "Hidden Layer")
st.plotly_chart(fig_hidden)

# Plot the output layer activations
fig_output = plot_activations(output_activation, "Output Layer")
st.plotly_chart(fig_output)

# Display the calculations
st.subheader("Forward Propagation Calculations")
st.write("Input Features (X):")
st.write(X)

st.write("Hidden Layer Weights (W1):")
st.write(hidden_weights)

st.write("Hidden Layer Bias (b1):")
st.write(hidden_bias)

st.write("Output Layer Weights (W2):")
st.write(output_weights)

st.write("Output Layer Bias (b2):")
st.write(output_bias)

st.write("Step 1: Calculate the weighted sum of inputs for the hidden layer (z1)")
z1 = np.dot(X, hidden_weights) + hidden_bias
st.write("z1 = X · W1 + b1")
st.write(z1)

st.write("Step 2: Apply the sigmoid activation function to the weighted sum (a1)")
a1 = sigmoid(z1)
st.write("a1 = sigmoid(z1)")
st.write(a1)

st.write("Step 3: Calculate the weighted sum of inputs for the output layer (z2)")
z2 = np.dot(a1, output_weights) + output_bias
st.write("z2 = a1 · W2 + b2")
st.write(z2)

st.write("Step 4: Apply the softmax activation function to the weighted sum (a2)")
a2 = softmax(z2)
st.write("a2 = softmax(z2)")
st.write(a2)

# Explanations
st.header("Forward Propagation Explained")
st.write("Forward propagation is the process of passing the input data through the neural network to obtain the predicted output.")
st.write("At each layer, the input data is transformed using the weights and biases, and an activation function is applied to introduce non-linearity.")
st.write("The last layer produces the final output of the network, which can be used for prediction or classification.")

st.subheader("Steps in Forward Propagation")
st.write("1. The input data is multiplied by the weights of the first layer and the bias is added.")
st.write("2. An activation function (e.g., sigmoid or softmax) is applied to the sum.")
st.write("3. The result is passed as input to the next layer, and steps 1-2 are repeated until the final layer.")
st.write("4. The output of the last layer gives the predictions of the neural network.")

st.subheader("Example")
st.write("In this example, we randomly generate input data and weights for a simple neural network with one hidden layer.")
st.write("You can adjust the number of input features, hidden neurons, output neurons, and the activation function using the sidebar controls.")
st.write("The activations of the hidden layer and output layer are visualized using bar plots, where each bar represents the activation of a neuron in that layer.")