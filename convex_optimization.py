import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Custom CSS and Page Configuration
st.set_page_config(layout="wide", page_title="Optimization Explorer", page_icon="📈")

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
def convex_func(x):
    return (x - 2)**2 + 1

def non_convex_func(x):
    return np.sin(x) + 0.5*x

def plot_function(x, y, title, current_x):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Function'))
    fig.add_trace(go.Scatter(x=[current_x], y=[y[np.argmin(np.abs(x - current_x))]], mode='markers', marker=dict(color='red', size=10), name='Current Point'))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='f(x)')
    return fig

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'x_convex': 0.0,
        'x_non_convex': 0.0
    }

def configure_sidebar():
    st.sidebar.header("Function Parameters")
    st.session_state.params['x_convex'] = st.sidebar.slider("x (Convex)", min_value=-5.0, max_value=5.0, value=st.session_state.params['x_convex'], step=0.1)
    st.session_state.params['x_non_convex'] = st.sidebar.slider("x (Non-Convex)", min_value=-5.0, max_value=5.0, value=st.session_state.params['x_non_convex'], step=0.1)

def convex_optimization():
    st.markdown("<h2 class='sub-header'>Convex Optimization</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Convex optimization involves a function that has only one minimum, corresponding to the global optimum. 
    There is no concept of local optima for convex optimization problems.
    </p>
    """, unsafe_allow_html=True)
    
    x = np.linspace(-5, 5, 100)
    y_convex = convex_func(x)
    fig_convex = plot_function(x, y_convex, 'Convex Function', st.session_state.params['x_convex'])
    st.plotly_chart(fig_convex, use_container_width=True)

def non_convex_optimization():
    st.markdown("<h2 class='sub-header'>Non-Convex Optimization</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Non-convex optimization involves a function that has multiple local optima. 
    Finding the global optimum in non-convex optimization can be challenging.
    </p>
    """, unsafe_allow_html=True)
    
    x = np.linspace(-5, 5, 100)
    y_non_convex = non_convex_func(x)
    fig_non_convex = plot_function(x, y_non_convex, 'Non-Convex Function', st.session_state.params['x_non_convex'])
    st.plotly_chart(fig_non_convex, use_container_width=True)

def learning_center():
    st.markdown("<h2 class='sub-header'>Optimization Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Optimization is the process of finding the best solution from all feasible solutions. In mathematical terms, it involves finding the minimum or maximum of a function.

    <h3>Key Concepts:</h3>
    1. <strong>Convex Optimization:</strong> Deals with convex functions which have a single global minimum. It's like finding the lowest point in a smooth bowl.
    2. <strong>Non-Convex Optimization:</strong> Involves functions with multiple local minima. It's like finding the lowest valley in a mountainous landscape.
    3. <strong>Global Optimum:</strong> The best solution among all possible solutions.
    4. <strong>Local Optimum:</strong> The best solution within a neighboring set of solutions.

    <h3>Applications:</h3>
    - Machine Learning: Training models often involves optimization (e.g., minimizing loss functions)
    - Finance: Portfolio optimization
    - Engineering: Designing efficient systems

    Experiment with the interactive demos to see how convex and non-convex functions behave!
    </p>
    """, unsafe_allow_html=True)

def quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main characteristic of a convex function in optimization?",
            "options": [
                "It has multiple local minima",
                "It has only one minimum, which is the global optimum",
                "It is always linear",
                "It cannot be optimized"
            ],
            "correct": 1,
            "explanation": "A convex function is like a smooth bowl - no matter where you start, you'll always roll down to the same lowest point (global minimum). This makes optimization much simpler and guarantees finding the best solution."
        },
        {
            "question": "Why can non-convex optimization be challenging?",
            "options": [
                "It's always impossible to solve",
                "It requires more computational power",
                "It may have multiple local optima, making it hard to find the global optimum",
                "It only applies to discrete problems"
            ],
            "correct": 2,
            "explanation": "Non-convex optimization is like navigating a hilly landscape. You might find yourself in a local valley (local optimum) and not know if it's the lowest point overall (global optimum). This makes it challenging to ensure you've found the best possible solution."
        },
        {
            "question": "What's the difference between a local optimum and a global optimum?",
            "options": [
                "Local optima only exist in convex functions",
                "Global optima are always harder to find",
                "A local optimum is the best in a neighborhood, while a global optimum is the best overall",
                "There is no difference"
            ],
            "correct": 2,
            "explanation": "Think of it like finding the best restaurant. A local optimum is the best restaurant in your neighborhood (local area), while the global optimum is the best restaurant in the entire city (overall problem space)."
        },
        {
            "question": "In which field is optimization commonly used?",
            "options": [
                "Only in pure mathematics",
                "Exclusively in computer graphics",
                "In various fields including machine learning, finance, and engineering",
                "Only in theoretical physics"
            ],
            "correct": 2,
            "explanation": "Optimization is like a Swiss Army knife - it's useful in many fields! In machine learning, it helps train models. In finance, it aids in portfolio management. In engineering, it helps design efficient systems. It's a versatile tool applicable to many real-world problems."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're an optimization expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering optimization concepts. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>📈 Optimization Explorer 📉</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

    configure_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Convex Optimization", "🔎 Non-Convex Optimization", "📚 Learning Center", "🎓 Quiz"])

    with tab1:
        convex_optimization()
    
    with tab2:
        non_convex_optimization()
    
    with tab3:
        learning_center()
    
    with tab4:
        quiz()

    st.markdown("""
    <div class='interpretation'>
    <p class='content-text'>
    Observe how the convex function has a single global minimum, while the non-convex function has multiple local optima. 
    The red dot indicates the current x-value selected by the slider. 
    Try moving the sliders to see how the point moves along the function!
    </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
