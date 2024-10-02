import streamlit as st
import random

st.set_page_config(page_title="Overfitting in Neural Networks Quiz", layout="wide")

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
    "Overfitting and Batch Normalization": [
        {
            "question": "Why are deep neural networks prone to overfitting?",
            "correct_answer": "They try to capture complex patterns in the data but may also capture noise",
            "incorrect_answer": "They are unable to capture complex patterns in the data",
            "explanation": "Deep neural networks are prone to overfitting because they try to capture complex patterns in the data but may also capture noise in the process."
        },
        {
            "question": "What is the goal of Batch Normalization?",
            "correct_answer": "To normalize the inputs of each layer to have a mean of zero and standard deviation of one",
            "incorrect_answer": "To increase the complexity of the neural network",
            "explanation": "Batch Normalization aims to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one."
        },
        {
            "question": "What is an advantage of using Batch Normalization?",
            "correct_answer": "It leads to faster learning rates",
            "incorrect_answer": "It increases the complexity of the model",
            "explanation": "Batch Normalization leads to faster learning rates since it ensures there's no activation value that's too high or too low, and allows each layer to learn independently of the others."
        }
    ],
    "Dropout": [
        {
            "question": "What problem does Dropout address in neural networks?",
            "correct_answer": "Co-dependency among neurons during training",
            "incorrect_answer": "Slow learning rates",
            "explanation": "Dropout addresses the problem of co-dependency among neurons during training, which can lead to overfitting of the training data."
        },
        {
            "question": "How does Dropout work?",
            "correct_answer": "It randomly shuts down some fraction of a layer's neurons at each training step",
            "incorrect_answer": "It adds more neurons to each layer",
            "explanation": "In Dropout, we randomly shut down some fraction of a layer's neurons at each training step by zeroing out the neuron values."
        },
        {
            "question": "What happens to the remaining neurons when applying Dropout?",
            "correct_answer": "Their values are multiplied by 1/(1-rd)",
            "incorrect_answer": "Their values remain unchanged",
            "explanation": "The remaining neurons have their values multiplied by 1/(1-rd), where rd is the dropout rate, so that the overall sum of the neuron values remains the same."
        }
    ],
    "Early Stopping": [
        {
            "question": "What is Early Stopping in neural networks?",
            "correct_answer": "A technique to stop training when validation performance starts worsening",
            "incorrect_answer": "A technique to increase the number of training iterations",
            "explanation": "Early Stopping is a technique similar to cross-validation, where training is stopped when the performance on the validation data starts worsening."
        },
        {
            "question": "How is overfitting usually reduced using Early Stopping?",
            "correct_answer": "By observing the training/validation accuracy gap during model training",
            "incorrect_answer": "By increasing the number of training iterations",
            "explanation": "Overfitting is usually reduced by observing the training/validation accuracy gap during model training and then stopping at the right point."
        },
        {
            "question": "In the figure shown, when does the model stop training?",
            "correct_answer": "At the dotted line",
            "incorrect_answer": "At the end of all iterations",
            "explanation": "In the given figure, the model will stop training at the dotted line since after that point, the model will start overfitting on the training data."
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
                "Understanding overfitting and its solutions is crucial for building effective neural networks!"
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
st.title("🧠 Overfitting in Neural Networks Quiz")
st.markdown("""
Welcome to the Overfitting in Neural Networks Quiz! Select a topic and test your knowledge on various techniques to prevent overfitting in deep learning. 
Good luck! 🍀
""")

# Topic selection dropdown
selected_topic = st.selectbox("Choose a topic:", list(quiz_topics.keys()))

# Run the quiz for the selected topic
run_quiz(selected_topic, quiz_topics[selected_topic])