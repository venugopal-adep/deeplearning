import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("Gradient Descent Visualization")
st.write("**Developed by : Venugopal Adep**")

# Define the objective function
def objective(x):
    return x ** 2

# Set the learning rate and number of iterations
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
num_iterations = st.sidebar.slider("Number of Iterations", min_value=1, max_value=50, value=10, step=1)

# Initialize the starting point
x = st.sidebar.slider("Starting Point", min_value=-5.0, max_value=5.0, value=5.0, step=0.1)

# Create lists to store the points for visualization
x_history = [x]
y_history = [objective(x)]

# Perform gradient descent
for _ in range(num_iterations):
    gradient = 2 * x
    x -= learning_rate * gradient
    x_history.append(x)
    y_history.append(objective(x))

# Create a trace for the objective function
x_vals = np.linspace(-5, 5, 100)
y_vals = objective(x_vals)
function_trace = go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Objective Function")

# Create a trace for the gradient descent steps
descent_trace = go.Scatter(
    x=x_history,
    y=y_history,
    mode="markers+lines",
    name="Gradient Descent",
    marker=dict(size=10, color="red"),
)

# Create the layout
layout = go.Layout(title="Gradient Descent", xaxis=dict(title="x"), yaxis=dict(title="f(x)"))

# Create the figure and display it using Streamlit
fig = go.Figure(data=[function_trace, descent_trace], layout=layout)
st.plotly_chart(fig)
