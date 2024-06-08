import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def activation_function(func_name, x):
    if func_name == 'Sigmoid':
        return sigmoid(x)
    elif func_name == 'Tanh':
        return tanh(x)
    elif func_name == 'ReLU':
        return relu(x)

st.set_page_config(page_title='Activation Functions', layout='wide')
st.title('Activation Functions in Neural Networks')

st.sidebar.header('Input Signals, Weights, and Bias')
num_inputs = st.sidebar.slider('Number of Input Signals', min_value=1, max_value=5, value=3, step=1)
inputs = [st.sidebar.number_input(f'Input Signal {i+1}', value=0.0) for i in range(num_inputs)]
weights = [st.sidebar.number_input(f'Weight {i+1}', value=1.0) for i in range(num_inputs)]
bias = st.sidebar.number_input('Bias', value=0.0)

st.sidebar.header('Activation Function')
func_name = st.sidebar.selectbox('Select Activation Function', ('Sigmoid', 'Tanh', 'ReLU'))

weighted_sum = sum(x * w for x, w in zip(inputs, weights)) + bias
activation_output = activation_function(func_name, weighted_sum)

st.subheader('Visualization')

x = np.linspace(-5, 5, 100)
y = activation_function(func_name, x)

trace1 = go.Scatter(x=x, y=y, mode='lines', name=func_name)
trace2 = go.Scatter(x=[weighted_sum], y=[activation_output], mode='markers', 
                    marker=dict(size=12, color='red'), name='Output')

layout = go.Layout(
    title=f'{func_name} Activation Function',
    xaxis=dict(title='Weighted Sum of Inputs'),
    yaxis=dict(title='Activation Output')
)

fig = go.Figure(data=[trace1, trace2], layout=layout)
st.plotly_chart(fig)

if func_name == 'Sigmoid':
    formula = r'$\sigma(x) = \frac{1}{1 + e^{-x}}$'
elif func_name == 'Tanh':
    formula = r'$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$'
else:  # ReLU
    formula = r'$\mathrm{ReLU}(x) = \max(0, x)$'

st.write(f'{func_name} Activation Function Formula: {formula}')

st.subheader('What is an Activation Function?')
st.write('''
An activation function is a crucial component in artificial neural networks that introduces non-linearity into the network's processing. It takes the weighted sum of inputs plus a bias term and transforms it into an output signal. The activation function determines whether a neuron should be activated ("fired") or not, based on whether the input to the neuron meets a certain threshold.

Activation functions are essential because they allow neural networks to learn and represent complex, non-linear relationships between inputs and outputs. Without activation functions, neural networks would be limited to representing linear relationships, which would severely restrict their ability to solve complex problems.

Some common activation functions include:

1. Sigmoid: Squashes the input to a value between 0 and 1, often used for binary classification.
2. Tanh (Hyperbolic Tangent): Similar to Sigmoid but outputs values between -1 and 1.
3. ReLU (Rectified Linear Unit): Outputs the input directly if it is positive, and 0 otherwise. It has become popular due to its simplicity and effectiveness.

Activation functions are used in various applications of neural networks, such as image classification, speech recognition, natural language processing, and many others. By introducing non-linearity, they enable neural networks to learn complex patterns and make accurate predictions or decisions based on input data.
''')

st.subheader('Result')
st.write(f'For input signals {inputs}, weights {weights}, and bias {bias}:')
st.write(f'Weighted Sum is {weighted_sum:.3f}')  
st.write(f'{func_name} Activation Output is {activation_output:.3f}')