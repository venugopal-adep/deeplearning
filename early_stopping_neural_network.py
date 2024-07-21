import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Custom CSS and Page Configuration
st.set_page_config(layout="wide", page_title="Early Stopping in Neural Networks Explorer", page_icon="🧠")

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
def generate_data():
    X, y = make_classification(n_samples=1000, n_classes=2, n_features=10, n_informative=8, random_state=np.random.randint(1000))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }

def train_model(X_train, y_train, X_val, y_val, early_stopping, patience):
    model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)

    train_scores = []
    val_scores = []
    best_val_score = 0
    counter = 0

    for epoch in range(500):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))

        train_score = accuracy_score(y_train, model.predict(X_train))
        val_score = accuracy_score(y_val, model.predict(X_val))

        train_scores.append(train_score)
        val_scores.append(val_score)

        if early_stopping:
            if val_score > best_val_score:
                best_val_score = val_score
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    return model, train_scores, val_scores

def plot_learning_curve(train_scores, val_scores, early_stopping_point=None):
    epochs = list(range(1, len(train_scores) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_scores, mode='lines', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=epochs, y=val_scores, mode='lines', name='Validation Accuracy'))

    if early_stopping_point:
        fig.add_shape(type='line', x0=early_stopping_point, y0=0, x1=early_stopping_point, y1=1,
                      line=dict(color='red', width=2, dash='dash'), name='Early Stopping')

    fig.update_layout(title='Learning Curve', xaxis_title='Epoch', yaxis_title='Accuracy')
    return fig

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'early_stopping': True,
        'patience': 5
    }
    st.session_state.data = generate_data()

def configure_sidebar():
    st.sidebar.header("Input Parameters")
    st.session_state.params['early_stopping'] = st.sidebar.checkbox("Enable Early Stopping", value=st.session_state.params['early_stopping'])
    st.session_state.params['patience'] = st.sidebar.slider("Patience", 1, 20, st.session_state.params['patience'])

    if st.sidebar.button("Regenerate Data"):
        st.session_state.data = generate_data()
        st.experimental_rerun()

def early_stopping_visualization():
    st.markdown("<h2 class='sub-header'>Early Stopping Visualization</h2>", unsafe_allow_html=True)
    
    # Train the model
    model, train_scores, val_scores = train_model(
        st.session_state.data['X_train'], st.session_state.data['y_train'],
        st.session_state.data['X_val'], st.session_state.data['y_val'],
        st.session_state.params['early_stopping'], st.session_state.params['patience']
    )

    # Find the early stopping point
    early_stopping_point = None
    if st.session_state.params['early_stopping']:
        early_stopping_point = len(val_scores) - st.session_state.params['patience']

    # Plot the learning curve
    fig = plot_learning_curve(train_scores, val_scores, early_stopping_point)
    st.plotly_chart(fig, use_container_width=True)

    # Evaluate the model
    train_accuracy = accuracy_score(st.session_state.data['y_train'], model.predict(st.session_state.data['X_train']))
    val_accuracy = accuracy_score(st.session_state.data['y_val'], model.predict(st.session_state.data['X_val']))
    test_accuracy = accuracy_score(st.session_state.data['y_test'], model.predict(st.session_state.data['X_test']))

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.write(f"Training Accuracy: {train_accuracy:.4f}")
    st.write(f"Validation Accuracy: {val_accuracy:.4f}")
    st.write(f"Test Accuracy: {test_accuracy:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
    st.write("The learning curve plot shows the training and validation accuracies over epochs.")
    st.write("- Blue line: Training accuracy")
    st.write("- Orange line: Validation accuracy")
    if st.session_state.params['early_stopping']:
        st.write("- Red dashed line: Early stopping point")
    st.write("Observe how the model's performance changes over time and how early stopping can prevent overfitting.")
    st.markdown("</div>", unsafe_allow_html=True)

def learning_center():
    st.markdown("<h2 class='sub-header'>Early Stopping Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Early stopping is a regularization technique used to prevent overfitting in neural networks. It involves monitoring the model's performance on a validation set during training and stopping the training process when the performance starts to degrade.

    <h3>How Early Stopping Works</h3>
    1. Split the data into training, validation, and test sets.
    2. Train the model on the training set and evaluate its performance on the validation set after each epoch.
    3. If the validation performance starts to worsen, keep track of the number of epochs without improvement.
    4. If the number of epochs without improvement reaches the specified patience, stop the training.
    5. Use the model with the best validation performance for final evaluation on the test set.

    <h3>Benefits of Early Stopping</h3>
    - Prevents overfitting by stopping the training process before the model starts to memorize the training data.
    - Helps to find the optimal number of training epochs automatically.
    - Reduces the training time by avoiding unnecessary epochs.

    <h3>Example</h3>
    In this demo, we train a neural network on a binary classification task and demonstrate the effect of early stopping.
    - Enable the 'Early Stopping' checkbox to apply early stopping during training.
    - Adjust the 'Patience' slider to set the number of epochs to wait for improvement before stopping.
    - Use the 'Regenerate Data' button to create a new random dataset and observe how the model performs on different data.
    The learning curve plot shows the training and validation accuracies over epochs. If early stopping is enabled, a vertical dashed line indicates the point at which training is stopped.
    </p>
    """, unsafe_allow_html=True)

def quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main purpose of early stopping in neural networks?",
            "options": [
                "To speed up the training process",
                "To prevent overfitting",
                "To increase the model's complexity",
                "To reduce the number of parameters in the model"
            ],
            "correct": 1,
            "explanation": "Early stopping is primarily used to prevent overfitting by stopping the training process when the model's performance on the validation set starts to degrade."
        },
        {
            "question": "What does the 'patience' parameter in early stopping refer to?",
            "options": [
                "The total number of training epochs",
                "The number of epochs to wait before starting early stopping",
                "The number of epochs with no improvement before stopping",
                "The learning rate of the model"
            ],
            "correct": 2,
            "explanation": "The 'patience' parameter in early stopping refers to the number of epochs to wait for improvement in the validation performance before stopping the training process."
        },
        {
            "question": "Which dataset is used to determine when to stop training in early stopping?",
            "options": [
                "Training set",
                "Validation set",
                "Test set",
                "All of the above"
            ],
            "correct": 1,
            "explanation": "The validation set is used to monitor the model's performance and determine when to stop training in early stopping. The training set is used for learning, and the test set is kept separate for final evaluation."
        },
        {
            "question": "What happens if the patience value is set too low in early stopping?",
            "options": [
                "The model will always overfit",
                "Training will take much longer",
                "The model might underfit due to insufficient training",
                "It has no effect on the training process"
            ],
            "correct": 2,
            "explanation": "If the patience value is set too low, the training might stop too early, leading to underfitting. The model might not have enough time to learn the underlying patterns in the data."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're an early stopping expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering early stopping. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>🧠 Early Stopping in Neural Networks Explorer 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

    configure_sidebar()

    tab1, tab2, tab3 = st.tabs(["🔍 Visualization", "📚 Learning Center", "🎓 Quiz"])

    with tab1:
        early_stopping_visualization()
    
    with tab2:
        learning_center()
    
    with tab3:
        quiz()

if __name__ == "__main__":
    main()
