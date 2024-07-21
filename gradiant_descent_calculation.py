import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Custom CSS and Page Configuration
st.set_page_config(layout="wide", page_title="Gradient Descent Explorer", page_icon="📉")

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
    return fig

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'x_init': 10.0,
        'learning_rate': 0.1,
        'num_iterations': 10
    }

def configure_sidebar():
    st.sidebar.header("Interactive Inputs")
    st.session_state.params['x_init'] = st.sidebar.number_input("Initial guess (x_init):", value=st.session_state.params['x_init'])
    st.session_state.params['learning_rate'] = st.sidebar.number_input("Learning rate (alpha):", value=st.session_state.params['learning_rate'])
    st.session_state.params['num_iterations'] = st.sidebar.number_input("Number of iterations:", value=st.session_state.params['num_iterations'], min_value=1, max_value=100, step=1)

def gradient_descent_visualization():
    st.markdown("<h2 class='sub-header'>Gradient Descent Visualization</h2>", unsafe_allow_html=True)
    
    if st.button("Run Gradient Descent"):
        x_history = gradient_descent(st.session_state.params['x_init'], st.session_state.params['learning_rate'], st.session_state.params['num_iterations'])
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write("Gradient Descent Steps:")
        for i, x in enumerate(x_history):
            st.write(f"Iteration {i}: x = {x:.4f}, f'(x) = {f_prime(x):.4f}, f(x) = {f(x):.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
        st.write("Formulas Used:")
        st.latex(r"f(x) = x^2")
        st.latex(r"f'(x) = 2x")
        st.latex(r"x_{new} = x_{old} - \alpha \cdot f'(x_{old})")
        st.markdown("</div>", unsafe_allow_html=True)
        
        fig = plot_function(x_history)
        st.plotly_chart(fig, use_container_width=True)

def learning_center():
    st.markdown("<h2 class='sub-header'>Gradient Descent Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Gradient Descent is an optimization algorithm used to find the minimum of a function. In this demo, we're minimizing the quadratic function f(x) = x^2.

    <h3>Key Concepts:</h3>
    1. <strong>Objective Function:</strong> f(x) = x^2. This is the function we want to minimize.
    2. <strong>Gradient:</strong> f'(x) = 2x. This is the derivative of our objective function.
    3. <strong>Learning Rate (α):</strong> Determines the size of steps we take in each iteration.
    4. <strong>Update Rule:</strong> x_new = x_old - α * f'(x_old). This is how we update our position in each iteration.

    <h3>How it works:</h3>
    1. Start at an initial point (x_init).
    2. Calculate the gradient at that point.
    3. Move in the opposite direction of the gradient (because we're minimizing).
    4. Repeat steps 2-3 for a set number of iterations.

    <h3>Interpretation:</h3>
    - The blue line represents our objective function f(x) = x^2.
    - The red dots show the path of gradient descent.
    - Watch how the algorithm converges towards the minimum (x = 0) as iterations increase.

    Experiment with different initial points, learning rates, and iterations to see how they affect the optimization process!
    </p>
    """, unsafe_allow_html=True)

def quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the minimum point of the function f(x) = x^2?",
            "options": [
                "x = 1",
                "x = -1",
                "x = 0",
                "x = 2"
            ],
            "correct": 2,
            "explanation": "The minimum point of f(x) = x^2 is at x = 0. This is where the parabola reaches its lowest point, touching the x-axis."
        },
        {
            "question": "What happens if we use a very large learning rate in this example?",
            "options": [
                "The algorithm will converge faster",
                "The algorithm may overshoot and fail to converge",
                "The function will change",
                "Nothing will change"
            ],
            "correct": 1,
            "explanation": "With a very large learning rate, the algorithm might overshoot the minimum point, potentially bouncing back and forth or even diverging away from the minimum."
        },
        {
            "question": "Why do we use the negative of the gradient in the update rule?",
            "options": [
                "To make the calculations easier",
                "To move towards the maximum of the function",
                "To move towards the minimum of the function",
                "It's just a convention"
            ],
            "correct": 2,
            "explanation": "We use the negative of the gradient because we want to move towards the minimum of the function. The gradient points in the direction of steepest increase, so we move in the opposite direction to decrease."
        },
        {
            "question": "What would happen if we started gradient descent at x = 0 for this function?",
            "options": [
                "It would immediately find the minimum",
                "It would move away from the minimum",
                "It would stay at x = 0",
                "It would throw an error"
            ],
            "correct": 2,
            "explanation": "If we start at x = 0, which is already the minimum of f(x) = x^2, the gradient would be zero (f'(0) = 2*0 = 0). Therefore, the algorithm would stay at x = 0, as there's no direction to move."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a gradient descent expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering gradient descent. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>📉 Gradient Descent Explorer 📉</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)
    st.write("This app demonstrates gradient descent to find the minimum of the quadratic function f(x) = x^2.")

    configure_sidebar()

    tab1, tab2, tab3 = st.tabs(["🔍 Visualization", "📚 Learning Center", "🎓 Quiz"])

    with tab1:
        gradient_descent_visualization()
    
    with tab2:
        learning_center()
    
    with tab3:
        quiz()

if __name__ == "__main__":
    main()
