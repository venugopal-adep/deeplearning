import streamlit as st
import plotly.graph_objects as go
import numpy as np

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
def objective(x):
    return x ** 2

def gradient_descent(x_start, learning_rate, num_iterations):
    x = x_start
    x_history = [x]
    y_history = [objective(x)]
    
    for _ in range(num_iterations):
        gradient = 2 * x
        x -= learning_rate * gradient
        x_history.append(x)
        y_history.append(objective(x))
    
    return x_history, y_history

def plot_gradient_descent(x_history, y_history):
    x_vals = np.linspace(-5, 5, 100)
    y_vals = objective(x_vals)
    
    function_trace = go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Objective Function")
    descent_trace = go.Scatter(
        x=x_history,
        y=y_history,
        mode="markers+lines",
        name="Gradient Descent",
        marker=dict(size=10, color="red"),
    )
    
    layout = go.Layout(title="Gradient Descent", xaxis=dict(title="x"), yaxis=dict(title="f(x)"))
    fig = go.Figure(data=[function_trace, descent_trace], layout=layout)
    
    return fig

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'learning_rate': 0.1,
        'num_iterations': 10,
        'x_start': 5.0
    }

def configure_sidebar():
    st.sidebar.header("Gradient Descent Parameters")
    st.session_state.params['learning_rate'] = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=st.session_state.params['learning_rate'], step=0.01)
    st.session_state.params['num_iterations'] = st.sidebar.slider("Number of Iterations", min_value=1, max_value=50, value=st.session_state.params['num_iterations'], step=1)
    st.session_state.params['x_start'] = st.sidebar.slider("Starting Point", min_value=-5.0, max_value=5.0, value=st.session_state.params['x_start'], step=0.1)

def gradient_descent_visualization():
    st.markdown("<h2 class='sub-header'>Gradient Descent Visualization</h2>", unsafe_allow_html=True)
    
    x_history, y_history = gradient_descent(
        st.session_state.params['x_start'],
        st.session_state.params['learning_rate'],
        st.session_state.params['num_iterations']
    )
    
    fig = plot_gradient_descent(x_history, y_history)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class='interpretation'>
    <p class='content-text'>
    The graph above shows the objective function (blue line) and the path taken by gradient descent (red markers).
    Each red marker represents one iteration of the algorithm. Observe how the algorithm converges towards the minimum of the function.
    </p>
    </div>
    """, unsafe_allow_html=True)

def learning_center():
    st.markdown("<h2 class='sub-header'>Gradient Descent Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Gradient Descent is an optimization algorithm used to find the minimum of a function. It's widely used in machine learning for training models.

    <h3>Key Concepts:</h3>
    1. <strong>Objective Function:</strong> The function we want to minimize. In this demo, it's f(x) = x².
    2. <strong>Gradient:</strong> The slope of the function at a given point. It indicates the direction of steepest ascent.
    3. <strong>Learning Rate:</strong> The step size for each iteration. It determines how big of a step we take in the direction of the negative gradient.
    4. <strong>Iterations:</strong> The number of times we update our position.

    <h3>How it works:</h3>
    1. Start at an initial point.
    2. Calculate the gradient at that point.
    3. Move in the opposite direction of the gradient (because we're minimizing).
    4. Repeat steps 2-3 for a set number of iterations or until convergence.

    <h3>Applications:</h3>
    - Training neural networks
    - Optimizing loss functions in various machine learning algorithms
    - Solving systems of equations

    Experiment with the interactive demo to see how different parameters affect the optimization process!
    </p>
    """, unsafe_allow_html=True)

def quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What does the gradient represent in gradient descent?",
            "options": [
                "The value of the function at a point",
                "The direction of steepest descent",
                "The learning rate",
                "The number of iterations"
            ],
            "correct": 1,
            "explanation": "The gradient points in the direction of steepest ascent. In gradient descent, we move in the opposite direction to find the minimum. It's like a ball rolling down a hill - it naturally moves in the direction of steepest descent."
        },
        {
            "question": "What happens if the learning rate is too large?",
            "options": [
                "The algorithm will converge faster",
                "The algorithm may overshoot the minimum and fail to converge",
                "The algorithm will always find the global minimum",
                "The gradient will become zero"
            ],
            "correct": 1,
            "explanation": "If the learning rate is too large, it's like taking too big steps while walking downhill. You might overshoot and end up going back and forth across the valley, or even climb up the other side!"
        },
        {
            "question": "What is the purpose of the objective function in gradient descent?",
            "options": [
                "To determine the learning rate",
                "To calculate the gradient",
                "To provide the value we want to minimize or maximize",
                "To set the number of iterations"
            ],
            "correct": 2,
            "explanation": "The objective function is like the landscape we're navigating. In optimization, we're trying to find the lowest point (for minimization) or highest point (for maximization) in this landscape."
        },
        {
            "question": "In the context of machine learning, what does gradient descent often minimize?",
            "options": [
                "The number of features",
                "The learning rate",
                "The loss or cost function",
                "The number of training examples"
            ],
            "correct": 2,
            "explanation": "In machine learning, gradient descent is often used to minimize the loss or cost function. This function measures how well our model is performing - the lower the loss, the better the model. It's like trying to find the bottom of a valley where the altitude represents the error of our model."
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
