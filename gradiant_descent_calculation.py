import streamlit as st
import numpy as np
import plotly.graph_objects as go

def f(x):
    return x**2

def f_prime(x):
    return 2*x

def gradient_descent(x_init, learning_rate, num_iterations):
    x = x_init
    x_history = [x]

    for _ in range(num_iterations):
        x = x - learning_rate * f_prime(x)
        x_history.append(x)

    return x_history

def plot_function(x_history):
    x = np.linspace(-10, 10, 100)
    y = f(x)

    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='f(x) = x^2'))
    fig.add_trace(go.Scatter(x=x_history, y=f(np.array(x_history)), mode='markers', marker=dict(color='red'), name='Gradient Descent'))
    fig.update_layout(title='Gradient Descent on f(x) = x^2', xaxis_title='x', yaxis_title='f(x)')
    st.plotly_chart(fig)

def main():
    st.title("Gradient Descent Demo")
    st.write('**Developed by : Venugopal Adep**')
    st.write("This app demonstrates gradient descent to find the minimum of the quadratic function f(x) = x^2.")

    st.sidebar.header("Interactive Inputs")
    x_init = st.sidebar.number_input("Initial guess (x_init):", value=10.0)
    learning_rate = st.sidebar.number_input("Learning rate (alpha):", value=0.1)
    num_iterations = st.sidebar.number_input("Number of iterations:", value=10, min_value=1, max_value=100, step=1)

    if st.sidebar.button("Run Gradient Descent"):
        x_history = gradient_descent(x_init, learning_rate, num_iterations)
        st.write("Gradient Descent Steps:")
        for i, x in enumerate(x_history):
            st.write(f"Iteration {i}: x = {x:.4f}, f'(x) = {f_prime(x):.4f}, f(x) = {f(x):.4f}")
        
        st.write("Formulas Used:")
        st.latex(r"f(x) = x^2")
        st.latex(r"f'(x) = 2x")
        st.latex(r"x_{new} = x_{old} - \alpha \cdot f'(x_{old})")
        
        plot_function(x_history)

if __name__ == "__main__":
    main()