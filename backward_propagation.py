import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title='Back Propagation Demo', layout='wide')

st.title('Interactive Back Propagation Demo')

st.sidebar.header('Network Architecture')
n_input = st.sidebar.number_input('Number of Input Nodes', min_value=1, max_value=5, value=2, step=1)
n_hidden = st.sidebar.number_input('Number of Hidden Nodes', min_value=1, max_value=5, value=3, step=1)
n_output = st.sidebar.number_input('Number of Output Nodes', min_value=1, max_value=3, value=1, step=1)
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.5, step=0.01)

st.sidebar.header('Training Data')
data_func = st.sidebar.selectbox('Select Function', ['Sine', 'Cosine', 'Square'])

def generate_data(func, n_samples=100):
    X = np.random.rand(n_samples, n_input) * np.pi * 2
    if func == 'Sine':
        y = np.sin(X[:,0])
    elif func == 'Cosine':  
        y = np.cos(X[:,0])
    else:
        y = X[:,0]**2
    return X, y

X, y = generate_data(data_func)

st.sidebar.header('Training')
train_button = st.sidebar.button('Train Network')

w_ih = np.random.randn(n_input, n_hidden)
w_ho = np.random.randn(n_hidden, n_output)

def forward(x, w_ih, w_ho):
    h = np.maximum(0, x @ w_ih)  # ReLU activation
    y = h @ w_ho
    return y, h

def backward(x, y, y_pred, h, w_ih, w_ho):
    dy = (y_pred - y.reshape(-1, 1))
    dw_ho = h.T @ dy
    dh = dy @ w_ho.T
    dh[h<=0] = 0  # ReLU gradient
    dw_ih = x.T @ dh
    return dw_ih, dw_ho

col1, col2 = st.columns(2)

with col1:
    st.header('Concept Explanation')
    st.write('''
    Back propagation is a key algorithm that enables neural networks to learn from data. Here's a step-by-step explanation:
    
    1. **Forward Pass**: The input data is passed through the network. Each node applies a weight to the input, and the weighted sum is passed through an activation function. This process is repeated layer by layer until the output is produced.
    
    2. **Error Calculation**: The predicted output is compared to the actual target value. The difference between these two is the error. The goal is to minimize this error.
    
    3. **Backward Pass**: The error is then propagated backwards through the network. Each node receives a portion of the error proportional to its contribution to the output. This is used to update the weights of the node, so that the error is reduced next time for similar input.
    
    4. **Weight Update**: The weights are updated in the opposite direction of the error gradient, scaled by a learning rate. This ensures the network learns to produce the correct output for the given input.
    
    5. **Iteration**: Steps 1-4 are repeated for many training examples, gradually adjusting the weights until the network converges to a state where it can accurately map inputs to outputs.
    
    The interactive demo on the right allows you to see this process in action. Adjust the network architecture, select a function to learn, and watch how the error decreases as the network trains!
    ''')

with col2:
    st.header('Interactive Demo')

    loss_history = []
    if train_button:
        bar = st.progress(0)
        for i in range(100):
            y_pred, h = forward(X, w_ih, w_ho)
            loss = np.mean((y_pred - y.reshape(-1, 1))**2)
            loss_history.append(loss)
            dw_ih, dw_ho = backward(X, y, y_pred, h, w_ih, w_ho)
            w_ih -= learning_rate * dw_ih
            w_ho -= learning_rate * dw_ho
            bar.progress(i + 1)
        
        st.subheader('Error Over Training Iterations')
        fig = go.Figure(data=go.Scatter(y=loss_history))
        fig.update_layout(xaxis_title='Iteration', yaxis_title='Mean Squared Error')
        st.plotly_chart(fig)

        st.subheader('Predictions vs Actual')
        y_pred, _ = forward(X, w_ih, w_ho)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[:,0], y=y, name='Actual'))
        fig.add_trace(go.Scatter(x=X[:,0], y=y_pred.flatten(), name='Predicted'))
        fig.update_layout(xaxis_title='Input', yaxis_title='Output')
        st.plotly_chart(fig)

        st.write('Final error after training: ', loss_history[-1])