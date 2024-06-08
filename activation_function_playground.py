import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def plot_activation_function(func, func_name):
    x = np.linspace(-10, 10, 1000)
    y = func(x)
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(title=func_name, xaxis_title='Input', yaxis_title='Output')
    return fig

def main():
    st.title("Activation Function Playground")
    st.write('**Developed by : Venugopal Adep**')
    st.markdown("""
    Welcome to the Activation Function Playground! This interactive tool allows you to explore and understand different activation functions commonly used in neural networks.

    ## What are Activation Functions?
    Activation functions are mathematical functions that introduce non-linearity into neural networks. They determine the output of a neuron based on its input. Different activation functions have different properties and are used in various scenarios.

    ## How to Use the Tool
    1. Select an activation function from the dropdown menu below.
    2. Observe the plot of the selected activation function.
    3. Read the explanation and examples provided for each activation function.
    4. Experiment with different activation functions to understand their behavior.

    Let's dive in and explore the fascinating world of activation functions!
    """)

    activation_functions = {
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "ReLU": relu,
        "Leaky ReLU": leaky_relu,
        "ELU": elu
    }

    selected_function = st.sidebar.selectbox("Select an activation function:", list(activation_functions.keys()))

    fig = plot_activation_function(activation_functions[selected_function], selected_function)
    st.plotly_chart(fig)

    if selected_function == "Sigmoid":
        st.markdown("""
        ## Sigmoid Activation Function
        The sigmoid activation function squashes the input values to a range between 0 and 1. It is defined as:
        
        $\\sigma(x) = \\frac{1}{1 + e^{-x}}$

        ### Example
        - Input: 0.5
        - Output: 0.622459
        
        The sigmoid function is commonly used in binary classification problems, where the goal is to predict a probability between 0 and 1.
        """)
    elif selected_function == "Tanh":
        st.markdown("""
        ## Tanh Activation Function
        The tanh (hyperbolic tangent) activation function squashes the input values to a range between -1 and 1. It is defined as:
        
        $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$
        
        ### Example
        - Input: 0.5
        - Output: 0.462117
        
        The tanh function is often used in recurrent neural networks (RNNs) and can handle negative inputs better than the sigmoid function.
        """)
    elif selected_function == "ReLU":
        st.markdown("""
        ## ReLU Activation Function
        The ReLU (Rectified Linear Unit) activation function returns the input value if it is positive, and 0 otherwise. It is defined as:
        
        $\\text{ReLU}(x) = \\max(0, x)$
        
        ### Example
        - Input: 0.5
        - Output: 0.5
        - Input: -0.5
        - Output: 0
        
        ReLU is widely used in deep neural networks due to its simplicity and ability to alleviate the vanishing gradient problem.
        """)
    elif selected_function == "Leaky ReLU":
        st.markdown("""
        ## Leaky ReLU Activation Function
        The Leaky ReLU activation function is a variation of ReLU that allows a small negative slope for negative inputs. It is defined as:
        
        $\\text{LeakyReLU}(x) = \\begin{cases}
        x, & \\text{if } x > 0 \\\\
        \\alpha x, & \\text{otherwise}
        \\end{cases}$
        
        where $\\alpha$ is a small constant (e.g., 0.01).
        
        ### Example
        - Input: 0.5
        - Output: 0.5
        - Input: -0.5
        - Output: -0.005 (assuming $\\alpha$ = 0.01)
        
        Leaky ReLU addresses the "dying ReLU" problem by allowing gradients to flow for negative inputs.
        """)
    elif selected_function == "ELU":
        st.markdown("""
        ## ELU Activation Function
        The ELU (Exponential Linear Unit) activation function is similar to ReLU but has a smooth transition for negative inputs. It is defined as:
        
        $\\text{ELU}(x) = \\begin{cases}
        x, & \\text{if } x > 0 \\\\
        \\alpha (e^x - 1), & \\text{otherwise}
        \\end{cases}$
        
        where $\\alpha$ is a hyperparameter that controls the saturation for negative inputs.
        
        ### Example
        - Input: 0.5
        - Output: 0.5
        - Input: -0.5
        - Output: -0.393469 (assuming $\\alpha$ = 1)
        
        ELU helps alleviate the vanishing gradient problem and can lead to faster learning and better generalization compared to ReLU.
        """)

if __name__ == "__main__":
    main()