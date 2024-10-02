import streamlit as st
import random

st.set_page_config(page_title="Optimization Techniques Quiz", layout="wide")

# Custom CSS (same as before)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f4f8;
        color: #1e1e1e;
    }
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
        text-align: center;
    }
    .question-box {
        background-color: #e0f2fe;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 30px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .score-box {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        background-color: #d5f5e3;
        color: #27ae60;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feedback-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feedback-correct {
        background-color: #d4edda;
        color: #155724;
    }
    .feedback-incorrect {
        background-color: #f8d7da;
        color: #721c24;
    }
    .explanation-box {
        background-color: #e9ecef;
        border-left: 5px solid #3498db;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Quiz questions based on the new images
quiz_topics = {
    "Types of Optimization": [
        {
            "question": "How many types of optimizations are mentioned?",
            "correct_answer": "Two",
            "incorrect_answer": "Three",
            "explanation": "Two types of optimizations: Convex optimization and Non-Convex optimization."
        },
        {
            "question": "What characteristic does a convex optimization function have?",
            "correct_answer": "It has only one minima",
            "incorrect_answer": "It has multiple minima",
            "explanation": "Convex optimization involves a function that has only one minima, corresponding to the Global optimum."
        },
        {
            "question": "Is there a concept of local optima for convex optimization problems?",
            "correct_answer": "No",
            "incorrect_answer": "Yes",
            "explanation": "There is no concept of local optima for convex optimization problems."
        }
    ],
    "Non-Convex Optimization": [
        {
            "question": "What does non-convex optimization involve?",
            "correct_answer": "A function with multiple optima values",
            "incorrect_answer": "A function with only one optimum value",
            "explanation": "Non-convex optimization involves a function that has multiple optima values, out of which only one is the global optima."
        },
        {
            "question": "How many global optima does a non-convex function typically have?",
            "correct_answer": "One",
            "incorrect_answer": "Multiple",
            "explanation": "In non-convex optimization, out of the multiple optima values, only one is the global optima."
        },
        {
            "question": "What can make it difficult to locate global optima in non-convex optimization?",
            "correct_answer": "The loss function",
            "incorrect_answer": "The input data",
            "explanation": "Depending on the loss function, it can be very difficult to locate global optima in non-convex optimization."
        }
    ],
    "Gradient Descent Algorithm": [
        {
            "question": "What is the goal of optimization according?",
            "correct_answer": "To find a set of weights that minimizes the loss function",
            "incorrect_answer": "To maximize the loss function",
            "explanation": "The goal of optimization is to find a set of weights that minimizes the loss function."
        },
        {
            "question": "What do optimization functions usually calculate?",
            "correct_answer": "The gradient",
            "incorrect_answer": "The loss",
            "explanation": "Optimization functions usually calculate the gradient, i.e., the partial derivative of the loss function with respect to weights."
        },
        {
            "question": "What is the learning rate in gradient descent?",
            "correct_answer": "A hyperparameter that determines the step size",
            "incorrect_answer": "The final value of the loss function",
            "explanation": "The learning rate is a hyperparameter which determines the step size (the amount by which the weights are updated)."
        }
    ],
    "Need for Advanced Optimizers": [
        {
            "question": "What type of functions do most real-life problems have?",
            "correct_answer": "Non-convex functions",
            "incorrect_answer": "Convex functions",
            "explanation": "In most real-life problems, we only have Non-convex functions."
        },
        {
            "question": "How many problems are associated with using Gradient Descent for non-convex functions?",
            "correct_answer": "Two",
            "incorrect_answer": "Three",
            "explanation": "There are two problems associated with using Gradient Descent for non-convex functions: slow convergence and getting stuck in local optima."
        },
        {
            "question": "Which of these is NOT mentioned as a variation of Gradient Descent or Optimizer?",
            "correct_answer": "Gaussian Descent",
            "incorrect_answer": "Stochastic Gradient Descent",
            "explanation": "Gaussian Descent is not mentioned in the list of Gradient Descent variations or Optimizers. The list includes Stochastic Gradient Descent, Mini Batch Gradient Descent, Stochastic Gradient Descent with Momentum, RMSprop, and Adam."
        }
    ],
    "Gradient Descent Variations": [
        {
            "question": "Which variation of Gradient Descent updates parameters by calculating gradients of the whole dataset?",
            "correct_answer": "Batch Gradient Descent",
            "incorrect_answer": "Stochastic Gradient Descent",
            "explanation": "Batch Gradient Descent updates the parameter by calculating gradients of the whole dataset."
        },
        {
            "question": "What is an advantage of Stochastic Gradient Descent over Batch Gradient Descent?",
            "correct_answer": "It is faster in learning",
            "incorrect_answer": "It is computationally efficient",
            "explanation": "Stochastic Gradient Descent is faster in learning than batch gradient descent and gives immediate performance insights."
        },
        {
            "question": "What is a disadvantage of Mini Batch Gradient Descent?",
            "correct_answer": "It adds up one more hyperparameter to tune",
            "incorrect_answer": "It learns very slowly",
            "explanation": "A disadvantage of Mini Batch Gradient Descent is that it adds up one more hyperparameter to tune."
        }
    ]
}

# Function to run the quiz (same as before)
def run_quiz(topic, questions):
    if f'game_state_{topic}' not in st.session_state:
        st.session_state[f'game_state_{topic}'] = {
            'questions': questions.copy(),
            'score': 0,
            'total_questions': len(questions),
            'current_question': None,
            'feedback': None,
            'explanation': None,
            'show_next': False
        }

    game_state = st.session_state[f'game_state_{topic}']

    if game_state['questions']:
        if not game_state['current_question']:
            game_state['current_question'] = random.choice(game_state['questions'])
            game_state['feedback'] = None
            game_state['explanation'] = None
            game_state['show_next'] = False

        # Display live score
        st.markdown(f"""
        <div class="score-box">
            🏆 Score: {game_state['score']} / {game_state['total_questions']}
        </div>
        """, unsafe_allow_html=True)

        # Display the question
        st.markdown(f"""
        <div class="question-box">
            <h3>Question:</h3>
            <p style="font-size: 18px; font-weight: 600;">{game_state['current_question']['question']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Buttons for selection
        if not game_state['show_next']:
            col1, col2 = st.columns(2)
            with col1:
                option1 = st.button(game_state['current_question']['correct_answer'], key=f"{topic}_option1")
            with col2:
                option2 = st.button(game_state['current_question']['incorrect_answer'], key=f"{topic}_option2")

            # Check answer and provide feedback
            if option1 or option2:
                selected_answer = game_state['current_question']['correct_answer'] if option1 else game_state['current_question']['incorrect_answer']
                correct_answer = game_state['current_question']['correct_answer']
                
                if selected_answer == correct_answer:
                    game_state['score'] += 1
                    game_state['feedback'] = "✅ Correct! Well done!"
                else:
                    game_state['feedback'] = f"❌ Oops! The correct answer is: {correct_answer}"
                
                game_state['explanation'] = game_state['current_question']['explanation']
                game_state['show_next'] = True
                st.rerun()

        # Display feedback, explanation, and next button
        if game_state['feedback']:
            st.markdown(f"""
            <div class="feedback-box {'feedback-correct' if '✅' in game_state['feedback'] else 'feedback-incorrect'}">
                {game_state['feedback']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="explanation-box">
                <h4>Explanation:</h4>
                <p>{game_state['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Next Question", key=f"{topic}_next_question"):
                game_state['questions'].remove(game_state['current_question'])
                game_state['current_question'] = None
                game_state['feedback'] = None
                game_state['explanation'] = None
                game_state['show_next'] = False
                st.rerun()

    else:
        final_score = game_state['score']
        total_questions = game_state['total_questions']
        percentage = (final_score / total_questions) * 100

        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background-color: #ffffff; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2>🎉 Congratulations! You've completed the {topic} Quiz! 🎉</h2>
            <p style="font-size: 24px; font-weight: 600; color: #2c3e50;">Your final score: {final_score} / {total_questions}</p>
            <p style="font-size: 20px; color: #3498db;">Accuracy: {percentage:.1f}%</p>
            <p style="font-size: 18px; font-style: italic; color: #7f8c8d; margin-top: 20px;">
                "Understanding optimization techniques is crucial for mastering machine learning algorithms!"
            </p>
        </div>
        """, unsafe_allow_html=True)

        if percentage == 100:
            st.balloons()

        if st.button("Play Again", key=f"{topic}_play_again"):
            st.session_state[f'game_state_{topic}'] = {
                'questions': questions.copy(),
                'score': 0,
                'total_questions': len(questions),
                'current_question': None,
                'feedback': None,
                'explanation': None,
                'show_next': False
            }
            st.rerun()

# Main app
st.title("🎯 Optimization Techniques Quiz")
st.markdown("""
Welcome to the Optimization Techniques Quiz! Select a topic and test your knowledge on various concepts related to optimization in machine learning. 
Good luck! 🍀
""")

# Topic selection dropdown
selected_topic = st.selectbox("Choose a topic:", list(quiz_topics.keys()))

# Run the quiz for the selected topic
run_quiz(selected_topic, quiz_topics[selected_topic])