import streamlit as st
import random

st.set_page_config(page_title="Neural Networks Quiz", layout="wide")

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
    "Neural Networks Basics": [
        {
            "question": "What are the three types of layers in an Artificial Neural Network?",
            "correct_answer": "Input layer, Hidden layer(s), Output layer",
            "incorrect_answer": "Input layer, Middle layer, Result layer",
            "explanation": "Artificial Neural Networks consist of three types of layers: Input layer, Hidden layer(s), and Output layer."
        },
        {
            "question": "What does the hidden layer represent in a neural network?",
            "correct_answer": "Intermediary nodes that divide the input space into regions with (soft) boundaries",
            "incorrect_answer": "The final output of the neural network",
            "explanation": "The hidden layer represents intermediary nodes that divide the input space into regions with (soft) boundaries."
        },
        {
            "question": "What does the output layer represent in a neural network?",
            "correct_answer": "The output of the neural network",
            "incorrect_answer": "The input dimensions of the data",
            "explanation": "The output layer represents the output of the neural network, which depends on the nature of the prediction task."
        }
    ],
    "Activation Functions": [
        {
            "question": "What are the three steps in which an artificial neural network works?",
            "correct_answer": "Multiply input signals with weights, add weighted signals, apply activation function",
            "incorrect_answer": "Collect data, process data, output results",
            "explanation": "An artificial neural network works in three steps: First, it multiplies input signals with corresponding weights; second, it adds the weighted signals together; third, it converts the result using an activation function."
        },
        {
            "question": "What is the purpose of the activation function in a neural network?",
            "correct_answer": "To act like a switch for the neuron",
            "incorrect_answer": "To collect input data",
            "explanation": "The purpose of the activation function is to act like a switch for the neuron, determining whether it should be activated or not."
        },
        {
            "question": "Why is the activation function critical to the overall functioning of the neural network?",
            "correct_answer": "Without it, the network becomes equivalent to one single neuron",
            "incorrect_answer": "It determines the input data for the network",
            "explanation": "The activation function is critical because without it, the whole neural network will mathematically become equivalent to one single neuron."
        }
    ],
    "Types of Activation Functions": [
        {
            "question": "What is the range of the Sigmoid activation function?",
            "correct_answer": "0 to 1",
            "incorrect_answer": "-1 to 1",
            "explanation": "The Sigmoid activation function has a range of 0 to 1 and is useful for giving probabilities."
        },
        {
            "question": "Which activation function is more computationally efficient?",
            "correct_answer": "ReLU",
            "incorrect_answer": "Sigmoid",
            "explanation": "The ReLU (Rectified Linear Unit) function is less computationally expensive compared to Sigmoid and Tanh."
        },
        {
            "question": "What is the range of the Tanh activation function?",
            "correct_answer": "-1 to 1",
            "incorrect_answer": "0 to 1",
            "explanation": "The Tanh activation function has a range of -1 to 1 and is steeper than the Sigmoid function."
        }
    ],
    "Forward Propagation": [
        {
            "question": "In forward propagation, how does the input data move through the network?",
            "correct_answer": "From input layer through hidden layer to output layer",
            "incorrect_answer": "From output layer through hidden layer to input layer",
            "explanation": "In forward propagation, the input data is propagated forward from the input layer through the hidden layer until it reaches the final/output layer where predictions are made."
        },
        {
            "question": "What are the three steps of data transformation in every neuron during forward propagation?",
            "correct_answer": "Sum weighted input, apply activation function, pass result to next layer",
            "incorrect_answer": "Collect data, process data, output results",
            "explanation": "At every layer, data gets transformed in three steps in every neuron: Sum weighted input, apply the activation function on the sum, and pass the result to all the neurons in the next layer."
        },
        {
            "question": "What type of function might be used in the output layer for binary classification?",
            "correct_answer": "Sigmoid function",
            "incorrect_answer": "ReLU function",
            "explanation": "The last layer (output layer) may have a sigmoid function for binary classification or a softmax function for multi-class classification."
        }
    ],
    "Back Propagation": [
        {
            "question": "What is the main purpose of back propagation in neural networks?",
            "correct_answer": "To re-calibrate the weights at every layer and node to minimize error",
            "incorrect_answer": "To propagate the input data forward through the network",
            "explanation": "Back propagation is the process of learning that the neural network employs to re-calibrate the weights at every layer and every node to minimize the error in the output layer."
        },
        {
            "question": "What happens to the weights during the first pass of forward propagation?",
            "correct_answer": "They are random numbers",
            "incorrect_answer": "They are optimized values",
            "explanation": "During the first pass of forward propagation, the weights are random numbers."
        },
        {
            "question": "What is the goal of back propagation?",
            "correct_answer": "To adjust weights in proportion to the error contribution",
            "incorrect_answer": "To increase the error in the output layer",
            "explanation": "The goal of back propagation is to adjust weights in proportion to the error contribution and in an iterative process identify the optimal combination of weights."
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
                "Understanding neural networks is key to unlocking the potential of artificial intelligence!"
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
st.title("🧠 Neural Networks Quiz")
st.markdown("""
Welcome to the Neural Networks Quiz! Select a topic and test your knowledge on various concepts related to neural networks and deep learning. 
Good luck! 🍀
""")

# Topic selection dropdown
selected_topic = st.selectbox("Choose a topic:", list(quiz_topics.keys()))

# Run the quiz for the selected topic
run_quiz(selected_topic, quiz_topics[selected_topic])