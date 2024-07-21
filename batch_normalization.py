import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Custom CSS and Page Configuration
st.set_page_config(layout="wide", page_title="Batch Normalization Explorer", page_icon="🧠")

st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #4B0082; text-align: center; margin-bottom: 2rem; text-shadow: 2px 2px 4px #cccccc;}
    .sub-header {font-size: 2rem; color: #8A2BE2; margin: 1.5rem 0;}
    .content-text {font-size: 1.1rem; line-height: 1.6;}
    .highlight {background-color: #E6E6FA; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;}
    .interpretation {background-color: #F0E6FF; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 5px solid #8A2BE2;}
    .explanation {background-color: #E6E6FA; padding: 0.8rem; border-radius: 5px; margin-top: 0.8rem;}
    .quiz-question {background-color: #F0E6FF; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 5px solid #8A2BE2;}
    .stButton>button {
        background-color: #9370DB; color: white; font-size: 1rem; padding: 0.5rem 1rem;
        border: none; border-radius: 4px; cursor: pointer; transition: all 0.3s;
    }
    .stButton>button:hover {background-color: #8A2BE2; transform: scale(1.05);}
</style>
""", unsafe_allow_html=True)

# Helper functions
def normalize_batch(batch, gamma, beta):
    mean = np.mean(batch, axis=0)
    var = np.var(batch, axis=0)
    normalized_batch = (batch - mean) / np.sqrt(var + 1e-8)
    output = gamma * normalized_batch + beta
    return output

def plot_distributions(before, after):
    x_before = np.linspace(np.min(before), np.max(before), 100)
    x_after = np.linspace(np.min(after), np.max(after), 100)

    pdf_before = norm.pdf(x_before, loc=np.mean(before), scale=np.std(before))
    pdf_after = norm.pdf(x_after, loc=np.mean(after), scale=np.std(after))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_before, y=pdf_before, mode='lines', name='Before Normalization'))
    fig.add_trace(go.Scatter(x=x_after, y=pdf_after, mode='lines', name='After Normalization'))

    fig.add_shape(type='line', x0=np.mean(before), y0=0, x1=np.mean(before), y1=np.max(pdf_before),
                  line=dict(color='blue', width=2, dash='dash'), name='Mean (Before)')
    fig.add_shape(type='line', x0=np.mean(after), y0=0, x1=np.mean(after), y1=np.max(pdf_after),
                  line=dict(color='red', width=2, dash='dash'), name='Mean (After)')

    fig.update_layout(title='Probability Distributions',
                      xaxis_title='Activation',
                      yaxis_title='Probability Density',
                      showlegend=True)

    return fig

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'num_samples': 100,
        'num_neurons': 10,
        'gamma': 1.0,
        'beta': 0.0
    }

def configure_sidebar():
    st.sidebar.header("Input Parameters")
    st.session_state.params['num_samples'] = st.sidebar.slider("Number of samples", 1, 1000, st.session_state.params['num_samples'])
    st.session_state.params['num_neurons'] = st.sidebar.slider("Number of neurons", 1, 100, st.session_state.params['num_neurons'])
    st.session_state.params['gamma'] = st.sidebar.slider("Gamma", 0.1, 2.0, st.session_state.params['gamma'], 0.1)
    st.session_state.params['beta'] = st.sidebar.slider("Beta", -2.0, 2.0, st.session_state.params['beta'], 0.1)

def batch_normalization_visualization():
    st.markdown("<h2 class='sub-header'>Batch Normalization Visualization</h2>", unsafe_allow_html=True)
    
    # Generate random batch of activations with a wider range
    hidden_before = np.random.normal(loc=2.0, scale=1.5, size=(st.session_state.params['num_samples'], st.session_state.params['num_neurons']))

    # Apply batch normalization
    hidden_after = normalize_batch(hidden_before, st.session_state.params['gamma'], st.session_state.params['beta'])

    # Plot the probability distributions before and after normalization
    fig = plot_distributions(hidden_before.flatten(), hidden_after.flatten())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
    st.write("The graph above shows the probability distributions of the activations before and after batch normalization.")
    st.write("- The blue curve represents the distribution before normalization.")
    st.write("- The red curve represents the distribution after normalization.")
    st.write("- The dashed lines indicate the mean of each distribution.")
    st.markdown("</div>", unsafe_allow_html=True)

def learning_center():
    st.markdown("<h2 class='sub-header'>Batch Normalization Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Batch Normalization is a technique used in deep neural networks to normalize the inputs of each layer. It helps to stabilize and speed up the training process.

    <h3>Why Batch Normalization?</h3>
    During training, the distribution of inputs to each layer can shift as the parameters of the previous layers change. This phenomenon is known as internal covariate shift. Batch Normalization addresses this issue by normalizing the inputs to have zero mean and unit variance.

    <h3>How Batch Normalization Works</h3>
    1. For each batch of data, compute the mean and variance of the activations.
    2. Normalize the activations by subtracting the mean and dividing by the square root of the variance.
    3. Scale and shift the normalized activations using learnable parameters gamma and beta.
    4. Use the normalized activations as inputs to the next layer.

    <h3>Benefits of Batch Normalization</h3>
    - Reduces internal covariate shift, allowing for faster learning.
    - Enables the use of higher learning rates, leading to faster convergence.
    - Reduces the sensitivity to weight initialization.
    - Acts as a regularizer, reducing the need for techniques like dropout.

    <h3>Example</h3>
    Consider a batch of activations from a hidden layer:
    </p>
    """, unsafe_allow_html=True)

    hidden_before = np.random.normal(loc=2.0, scale=1.5, size=(5, st.session_state.params['num_neurons']))
    hidden_after = normalize_batch(hidden_before, st.session_state.params['gamma'], st.session_state.params['beta'])

    st.write(hidden_before)
    st.write(f"After applying Batch Normalization with gamma={st.session_state.params['gamma']} and beta={st.session_state.params['beta']}:")
    st.write(hidden_after)
    st.write("Notice how the activations are normalized to have zero mean and unit variance, and then scaled and shifted by gamma and beta.")

def quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main purpose of Batch Normalization?",
            "options": [
                "To increase the complexity of the neural network",
                "To normalize the inputs of each layer, reducing internal covariate shift",
                "To reduce the number of parameters in the model",
                "To increase the training time of the model"
            ],
            "correct": 1,
            "explanation": "Batch Normalization normalizes the inputs of each layer, which helps reduce internal covariate shift. This stabilizes and speeds up the training process."
        },
        {
            "question": "What are the learnable parameters in Batch Normalization?",
            "options": [
                "Mean and variance",
                "Alpha and beta",
                "Gamma and beta",
                "Lambda and epsilon"
            ],
            "correct": 2,
            "explanation": "Gamma and beta are the learnable parameters in Batch Normalization. They allow the network to scale and shift the normalized values if needed."
        },
        {
            "question": "Which of the following is NOT a benefit of Batch Normalization?",
            "options": [
                "Faster learning",
                "Reduced sensitivity to weight initialization",
                "Increased model complexity",
                "Regularization effect"
            ],
            "correct": 2,
            "explanation": "Batch Normalization does not necessarily increase model complexity. Its main benefits include faster learning, reduced sensitivity to initialization, and a regularization effect."
        },
        {
            "question": "What does Batch Normalization do to the mean and variance of its inputs?",
            "options": [
                "Increases both mean and variance",
                "Decreases both mean and variance",
                "Sets mean to 0 and variance to 1",
                "Leaves them unchanged"
            ],
            "correct": 2,
            "explanation": "Batch Normalization normalizes its inputs to have a mean of 0 and a variance of 1 before applying the scale (gamma) and shift (beta) operations."
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
        st.markdown(f"<p class='sub-header'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Batch Normalization expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Batch Normalization. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>🧠 Batch Normalization Explorer 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

    configure_sidebar()

    tab1, tab2, tab3 = st.tabs(["🔍 Visualization", "📚 Learning Center", "🎓 Quiz"])

    with tab1:
        batch_normalization_visualization()
    
    with tab2:
        learning_center()
    
    with tab3:
        quiz()

if __name__ == "__main__":
    main()
