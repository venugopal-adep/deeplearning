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
st.write("**Developed by : Venugopal Adep**")

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

# Quiz Section
st.header("Neural Network Forward Propagation Quiz")
st.write("Test your understanding of forward propagation in neural networks with these simple questions!")

# Question 1
st.subheader("Question 1: What is forward propagation?")
q1_answer = st.checkbox("Click to reveal the answer", key="q1")
if q1_answer:
    st.write("""
    **Answer:** Forward propagation is like passing a baton in a relay race through a neural network. 

    - It starts with the input data (the first runner).
    - The data moves through each layer of the network (like different runners in the race).
    - At each layer, the data is transformed using weights and biases (like each runner's unique running style).
    - Finally, it reaches the output layer (the finish line), giving us the network's prediction.

    **Example:** Imagine you're using a neural network to recognize handwritten digits. The forward propagation takes the image of a digit (input), passes it through layers that detect features like lines and curves, and finally outputs a prediction of what digit it thinks the image represents.
    """)

# Question 2
st.subheader("Question 2: What is an activation function, and why is it important?")
q2_answer = st.checkbox("Click to reveal the answer", key="q2")
if q2_answer:
    st.write("""
    **Answer:** An activation function is like a decision-maker in each neuron of the network. It determines whether and how strongly a neuron should "fire" based on its input.

    Activation functions are important because:
    1. They introduce non-linearity, allowing the network to learn complex patterns.
    2. They help control the strength of the signal passing through the neuron.

    **Example:** Think of a neuron as a light switch with a dimmer. The activation function decides:
    - Whether to turn the light on (activate the neuron)
    - How bright the light should be (strength of activation)

    Without this, our neural network would just be doing simple linear calculations, like basic math, and couldn't learn complex patterns in data.
    """)

# Question 3
st.subheader("Question 3: What's the difference between the sigmoid and softmax activation functions?")
q3_answer = st.checkbox("Click to reveal the answer", key="q3")
if q3_answer:
    st.write("""
    **Answer:** Sigmoid and softmax are both activation functions, but they serve different purposes:

    **Sigmoid:**
    - Like a personal score between 0 and 1
    - Used for binary classification (yes/no decisions)
    - Each neuron's output is independent of others

    **Example:** Deciding if an email is spam (1) or not spam (0).

    **Softmax:**
    - Like splitting 100% among multiple choices
    - Used for multi-class classification
    - Outputs are probabilities that sum to 1 across all neurons

    **Example:** Classifying an image as a cat, dog, or bird. Softmax might output: 
      - Cat: 70%
      - Dog: 20%
      - Bird: 10%
    """)
