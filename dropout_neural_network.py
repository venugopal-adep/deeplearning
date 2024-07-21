import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Custom CSS and Page Configuration
st.set_page_config(layout="wide", page_title="Dropout in Neural Networks Explorer", page_icon="🧠")

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
def apply_dropout(neurons, dropout_rate):
    num_dropped = int(len(neurons) * dropout_rate)
    dropped_indices = np.random.choice(len(neurons), num_dropped, replace=False)
    neurons[dropped_indices] = 0
    return neurons

def plot_decision_boundary(model, X, y, dropout_rate=0.0):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    fig = go.Figure(data=go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=True))
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, size=10, line=dict(width=1, color='white'))))
    fig.update_layout(title=f'Decision Boundary (Dropout Rate: {dropout_rate})', xaxis_title='X1', yaxis_title='X2')
    return fig

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'dropout_rate': 0.0
    }
    # Generate dataset
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def configure_sidebar():
    st.sidebar.header("Input Parameters")
    st.session_state.params['dropout_rate'] = st.sidebar.slider("Dropout Rate", 0.0, 1.0, st.session_state.params['dropout_rate'], 0.1)

def dropout_visualization():
    st.markdown("<h2 class='sub-header'>Dropout Visualization</h2>", unsafe_allow_html=True)
    
    # Train the model
    model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(st.session_state.X_train, st.session_state.y_train)

    # Apply dropout to the trained model
    for layer in model.coefs_:
        layer *= np.random.binomial(1, 1 - st.session_state.params['dropout_rate'], size=layer.shape) / (1 - st.session_state.params['dropout_rate'])

    # Make predictions and calculate accuracy
    y_pred_train = model.predict(st.session_state.X_train)
    y_pred_test = model.predict(st.session_state.X_test)
    train_accuracy = accuracy_score(st.session_state.y_train, y_pred_train)
    test_accuracy = accuracy_score(st.session_state.y_test, y_pred_test)

    # Plot the decision boundary
    fig = plot_decision_boundary(model, np.vstack((st.session_state.X_train, st.session_state.X_test)), 
                                 np.hstack((st.session_state.y_train, st.session_state.y_test)), 
                                 st.session_state.params['dropout_rate'])
    st.plotly_chart(fig, use_container_width=True)

    # Display the accuracy scores
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.write(f"Training Accuracy: {train_accuracy:.2f}")
    st.write(f"Test Accuracy: {test_accuracy:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
    st.write("The plot above shows the decision boundary of the neural network on the moon dataset.")
    st.write("- Blue and red regions represent the areas classified as different classes.")
    st.write("- Points represent the actual data samples.")
    st.write("- Adjust the 'Dropout Rate' in the sidebar to see how it affects the decision boundary and model accuracy.")
    st.markdown("</div>", unsafe_allow_html=True)

def learning_center():
    st.markdown("<h2 class='sub-header'>Dropout Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Dropout is a regularization technique used in neural networks to prevent overfitting. It randomly sets a fraction of the neurons to zero during each training step, which helps to reduce co-dependency among neurons and improves generalization.

    <h3>How Dropout Works</h3>
    1. During each training step, randomly select a fraction (dropout rate) of neurons to be dropped out.
    2. Set the values of the dropped-out neurons to zero, effectively deactivating them.
    3. Train the network with the remaining active neurons.
    4. During inference (prediction), multiply the neuron values by (1 - dropout rate) to compensate for the scaling.

    <h3>Benefits of Dropout</h3>
    - Reduces overfitting by preventing neurons from relying too much on each other.
    - Forces the network to learn more robust features that are useful in conjunction with different random subsets of neurons.
    - Provides a way of approximately combining many different neural network architectures efficiently.

    <h3>Example</h3>
    In this demo, we train a neural network on a binary classification task (moon dataset) and visualize the effect of dropout on the decision boundary.
    Adjust the 'Dropout Rate' slider to see how dropout affects the network's performance and decision boundary.
    </p>
    """, unsafe_allow_html=True)

def quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main purpose of dropout in neural networks?",
            "options": [
                "To increase the complexity of the model",
                "To reduce overfitting and improve generalization",
                "To speed up the training process",
                "To increase the number of neurons in each layer"
            ],
            "correct": 1,
            "explanation": "Dropout is primarily used to reduce overfitting and improve the model's ability to generalize to new data. It does this by preventing neurons from becoming too dependent on each other."
        },
        {
            "question": "What happens to neurons during dropout?",
            "options": [
                "They are permanently removed from the network",
                "Their weights are set to zero",
                "They are temporarily deactivated by setting their output to zero",
                "Their activation function is changed"
            ],
            "correct": 2,
            "explanation": "During dropout, selected neurons are temporarily deactivated by setting their output to zero. This is done randomly for each training step, not permanently."
        },
        {
            "question": "How does dropout affect the network during inference (prediction)?",
            "options": [
                "It has no effect during inference",
                "Neurons are randomly dropped during inference",
                "All neurons are used, but their outputs are scaled",
                "Only the neurons that were never dropped during training are used"
            ],
            "correct": 2,
            "explanation": "During inference, all neurons are used, but their outputs are scaled by multiplying by (1 - dropout rate). This compensates for the fact that more neurons are active than during training."
        },
        {
            "question": "What is a potential drawback of using a very high dropout rate?",
            "options": [
                "It will always improve model performance",
                "It may lead to underfitting",
                "It will increase the training speed",
                "It will increase the model's capacity"
            ],
            "correct": 1,
            "explanation": "While dropout can help prevent overfitting, using a very high dropout rate may lead to underfitting. This is because the model might not have enough capacity to learn the underlying patterns in the data if too many neurons are dropped."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a dropout expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering dropout. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>🧠 Dropout in Neural Networks Explorer 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

    configure_sidebar()

    tab1, tab2, tab3 = st.tabs(["🔍 Visualization", "📚 Learning Center", "🎓 Quiz"])

    with tab1:
        dropout_visualization()
    
    with tab2:
        learning_center()
    
    with tab3:
        quiz()

if __name__ == "__main__":
    main()
