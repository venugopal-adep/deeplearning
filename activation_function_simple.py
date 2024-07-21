import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="Neural Network Activation Functions", page_icon="🧠")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #8A2BE2;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #9370DB;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8A2BE2;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #E6E6FA;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .quiz-question {
        background-color: #F0E6FA;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #8A2BE2;
    }
    .explanation {
        background-color: #E6F3FF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🧠 Activation Functions in Neural Networks 🧠</h1>", unsafe_allow_html=True)

# Functions
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

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Activation Function Explorer", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Activation Function Visualization and Calculation</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<h3 class='content-text'>Input Parameters</h3>", unsafe_allow_html=True)
        
        num_inputs = st.slider('Number of Input Signals', min_value=1, max_value=5, value=3, step=1)
        inputs = [st.number_input(f'Input Signal {i+1}', value=0.0, format="%.2f") for i in range(num_inputs)]
        weights = [st.number_input(f'Weight {i+1}', value=1.0, format="%.2f") for i in range(num_inputs)]
        bias = st.number_input('Bias', value=0.0, format="%.2f")

        func_name = st.selectbox('Select Activation Function', ('Sigmoid', 'Tanh', 'ReLU'))

        # Calculation
        weighted_sum = sum(x * w for x, w in zip(inputs, weights)) + bias
        activation_output = activation_function(func_name, weighted_sum)

        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3 class='content-text'>Calculation Results</h3>", unsafe_allow_html=True)
        st.write(f'Weighted Sum: {weighted_sum:.3f}')  
        st.write(f'{func_name} Activation Output: {activation_output:.3f}')
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 class='content-text'>Activation Function Visualization</h3>", unsafe_allow_html=True)
        
        x = np.linspace(-5, 5, 100)
        y = activation_function(func_name, x)
        trace1 = go.Scatter(x=x, y=y, mode='lines', name=func_name)
        trace2 = go.Scatter(x=[weighted_sum], y=[activation_output], mode='markers', 
                            marker=dict(size=12, color='red'), name='Output')
        layout = go.Layout(
            xaxis=dict(title='Weighted Sum of Inputs'),
            yaxis=dict(title='Activation Output')
        )
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        st.plotly_chart(fig, use_container_width=True)

        if func_name == 'Sigmoid':
            formula = r'$\sigma(x) = \frac{1}{1 + e^{-x}}$'
        elif func_name == 'Tanh':
            formula = r'$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$'
        else:  # ReLU
            formula = r'$\mathrm{ReLU}(x) = \max(0, x)$'
        st.markdown(f"<div class='content-text'><b>{func_name} Activation Function Formula:</b> {formula}</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Learn More About Activation Functions</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    An activation function is a crucial component in artificial neural networks that introduces non-linearity into the network's processing. It takes the weighted sum of inputs plus a bias term and transforms it into an output signal. The activation function determines whether a neuron should be activated ("fired") or not, based on whether the input to the neuron meets a certain threshold.

    Activation functions are essential because they allow neural networks to learn and represent complex, non-linear relationships between inputs and outputs. Without activation functions, neural networks would be limited to representing linear relationships, which would severely restrict their ability to solve complex problems.

    Some common activation functions include:
    1. Sigmoid: Squashes the input to a value between 0 and 1, often used for binary classification.
    2. Tanh (Hyperbolic Tangent): Similar to Sigmoid but outputs values between -1 and 1.
    3. ReLU (Rectified Linear Unit): Outputs the input directly if it is positive, and 0 otherwise. It has become popular due to its simplicity and effectiveness.

    Activation functions are used in various applications of neural networks, such as image classification, speech recognition, natural language processing, and many others. By introducing non-linearity, they enable neural networks to learn complex patterns and make accurate predictions or decisions based on input data.
    </p>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Test Your Activation Function Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main purpose of activation functions in neural networks?",
            "options": [
                "To speed up computations",
                "To introduce non-linearity",
                "To reduce the number of neurons",
                "To increase the number of layers"
            ],
            "correct": 1,
            "explanation": "Activation functions introduce non-linearity into neural networks, allowing them to learn and represent complex relationships between inputs and outputs."
        },
        {
            "question": "Which of the following is NOT a common activation function?",
            "options": ["Sigmoid", "Tanh", "ReLU", "Cosine"],
            "correct": 3,
            "explanation": "Sigmoid, Tanh, and ReLU are common activation functions. Cosine is not typically used as an activation function in neural networks."
        },
        {
            "question": "What is the output range of the Sigmoid activation function?",
            "options": ["0 to 1", "-1 to 1", "0 to infinity", "-infinity to infinity"],
            "correct": 0,
            "explanation": "The Sigmoid function squashes its input to a value between 0 and 1, making it useful for binary classification tasks."
        },
        {
            "question": "Which activation function is known for helping to mitigate the vanishing gradient problem?",
            "options": ["Sigmoid", "Tanh", "ReLU", "All of the above"],
            "correct": 2,
            "explanation": "ReLU (Rectified Linear Unit) is known for helping to mitigate the vanishing gradient problem because it doesn't saturate for positive inputs, allowing gradients to flow more easily during backpropagation."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<div class='quiz-question'>", unsafe_allow_html=True)
        st.markdown(f"<p class='content-text'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.markdown(f"<div class='explanation'><p>{q['explanation']}</p></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='tab-subheader'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're an activation function expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering activation functions. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

# Conclusion
st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
You've explored the world of activation functions in neural networks! Remember:

1. Activation functions introduce non-linearity into neural networks.
2. Common activation functions include Sigmoid, Tanh, and ReLU.
3. Each activation function has its own characteristics and use cases.
4. The choice of activation function can significantly impact a neural network's performance.
5. Visualizing activation functions helps in understanding their behavior.

Keep exploring and applying these concepts in your deep learning journey!
</p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: #FAFAFA;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class='footer'>
    Created with ❤️ by Your Name | © 2023 All Rights Reserved
</div>
""", unsafe_allow_html=True)
