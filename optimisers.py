import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

def function_to_optimize(x):
    return x ** 2

def derivative_function(x):
    return 2 * x

def batch_gradient_descent(lr, iterations, function, derivative, initial_point):
    x = initial_point
    for _ in range(iterations):
        grad = derivative(x)
        x -= lr * grad
        yield x

def stochastic_gradient_descent(lr, iterations, function, derivative, initial_point):
    x = initial_point
    for _ in range(iterations):
        grad = derivative(x)
        x -= lr * grad
        yield x

def mini_batch_gradient_descent(lr, iterations, function, derivative, initial_point, batch_size):
    x = initial_point
    for _ in range(iterations):
        grad = derivative(x)
        x -= lr * grad
        yield x

st.title('Gradient Descent Visualization Tool')

with st.sidebar:
    method = st.selectbox('Select Gradient Descent Method', ['Batch', 'Stochastic', 'Mini-Batch'])
    lr = st.slider('Learning Rate', 0.01, 1.0, 0.1, 0.01)
    iterations = st.slider('Number of Iterations', 1, 50, 10)
    initial_point = st.slider('Starting Point', -10.0, 10.0, 5.0)
    if method == 'Mini-Batch':
        batch_size = st.slider('Batch Size', 1, 10, 1)
    start_optimization = st.button('Start Optimization')

if start_optimization:
    x_vals = np.linspace(-10, 10, 400)
    y_vals = function_to_optimize(x_vals)
    points = [initial_point]
    for new_point in (batch_gradient_descent(lr, iterations, function_to_optimize, derivative_function, initial_point) if method == 'Batch'
                      else stochastic_gradient_descent(lr, iterations, function_to_optimize, derivative_function, initial_point) if method == 'Stochastic'
                      else mini_batch_gradient_descent(lr, iterations, function_to_optimize, derivative_function, initial_point, batch_size)):
        points.append(new_point)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Objective Function'))
        fig.add_trace(go.Scatter(x=points, y=[function_to_optimize(x) for x in points], mode='markers+lines', name='Optimization Path', marker=dict(color='red')))
        fig.update_layout(xaxis_title='X', yaxis_title='f(X)', title='Gradient Descent Progression')
        st.plotly_chart(fig, use_container_width=True)
        time.sleep(0.5)  # Delay to allow for visualization of the animation
