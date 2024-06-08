import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.write("## Activation Function Visualization")
st.write("**Developed by : Venugopal Adep**")

st.markdown("""
This tool visualizes different neural network activation functions. 
Use the checkboxes in the sidebar to toggle the display of the Sigmoid, Tanh, and ReLU functions. 
The graph updates automatically based on your selections.
""")

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    # Modify ReLU to cut off at y=1
    y = np.maximum(0, x)
    return np.where(y > 1, None, y)  # Returns None where y > 1

# Create a range of input values
x = np.linspace(-10, 10, 400)  # Increased resolution for smoother cutoff

# Create checkboxes for selecting activation functions
show_sigmoid = st.sidebar.checkbox("Show Sigmoid", value=True)
show_tanh = st.sidebar.checkbox("Show Tanh", value=True)
show_relu = st.sidebar.checkbox("Show ReLU", value=True)

# Create traces for selected activation functions
traces = []
if show_sigmoid:
    sigmoid_y = sigmoid(x)
    traces.append(go.Scatter(x=x, y=sigmoid_y, mode="lines", name="Sigmoid"))
if show_tanh:
    tanh_y = tanh(x)
    traces.append(go.Scatter(x=x, y=tanh_y, mode="lines", name="Tanh"))
if show_relu:
    relu_y = relu(x)
    traces.append(go.Scatter(x=x, y=relu_y, mode="lines", name="ReLU"))

# Create the layout
layout = go.Layout(
    title="Activation Functions",
    xaxis=dict(title="Input"),
    yaxis=dict(title="Output", range=[-1.5, 1.5]),  # Adjusted for visibility
    shapes=[
        # Line representing Y-axis
        dict(
            type='line',
            x0=0, y0=-1.5, x1=0, y1=1.5,
            line=dict(color='Black', width=2)
        )
    ]
)

# Create the figure and display it using Streamlit
fig = go.Figure(data=traces, layout=layout)
st.plotly_chart(fig)

st.markdown("""
### Conceptual Overview

**Sigmoid Function**: Maps any real-valued number into the (0, 1) interval, ideal for models where we need to predict the probability as an output since the probability of anything exists only between the range of 0 and 1.

**Tanh Function**: The hyperbolic tangent function, which shapes similar to the sigmoid but maps input values to a range between -1 and 1. It's useful when dealing with negative numbers.

**ReLU Function**: The Rectified Linear Activation Function is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.
""")
