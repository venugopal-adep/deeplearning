import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Custom CSS and Page Configuration
st.set_page_config(layout="wide", page_title="Neural Network Forward Propagation Explorer", page_icon="🧠")

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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_propagation(X, hidden_weights, hidden_bias, output_weights, output_bias, activation):
    hidden_layer = np.dot(X, hidden_weights) + hidden_bias
    hidden_activation = activation(hidden_layer)
    output_layer = np.dot(hidden_activation, output_weights) + output_bias
    if activation == softmax:
        output_activation = softmax(output_layer)
    else:
        output_activation = sigmoid(output_layer)
    return hidden_activation, output_activation

def plot_activations(layer_activations, layer_name):
    fig = go.Figure(data=[go.Bar(x=list(range(len(layer_activations[0]))), y=layer_activations[0])])
    fig.update_layout(title=f"{layer_name} Activations", xaxis_title="Neuron", yaxis_title="Activation")
    return fig

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'num_features': 5,
        'num_hidden_neurons': 5,
        'num_output_neurons': 2,
        'activation_func': "Sigmoid"
    }

def configure_sidebar():
    st.sidebar.header("Input Parameters")
    st.session_state.params['num_features'] = st.sidebar.slider("Number of Input Features", 1, 10, st.session_state.params['num_features'])
    st.session_state.params['num_hidden_neurons'] = st.sidebar.slider("Number of Hidden Neurons", 1, 10, st.session_state.params['num_hidden_neurons'])
    st.session_state.params['num_output_neurons'] = st.sidebar.slider("Number of Output Neurons", 1, 10, st.session_state.params['num_output_neurons'])
    st.session_state.params['activation_func'] = st.sidebar.selectbox("Activation Function", ("Sigmoid", "Softmax"))

def generate_network_data():
    X = np.random.randn(1, st.session_state.params['num_features'])
    hidden_weights = np.random.randn(st.session_state.params['num_features'], st.session_state.params['num_hidden_neurons'])
    hidden_bias = np.random.randn(1, st.session_state.params['num_hidden_neurons'])
    output_weights = np.random.randn(st.session_state.params['num_hidden_neurons'], st.session_state.params['num_output_neurons'])
    output_bias = np.random.randn(1, st.session_state.params['num_output_neurons'])
    return X, hidden_weights, hidden_bias, output_weights, output_bias

def network_visualization():
    st.markdown("<h2 class='sub-header'>Network Visualization</h2>", unsafe_allow_html=True)

    X, hidden_weights, hidden_bias, output_weights, output_bias = generate_network_data()
    
    activation = sigmoid if st.session_state.params['activation_func'] == "Sigmoid" else softmax
    hidden_activation, output_activation = forward_propagation(X, hidden_weights, hidden_bias, output_weights, output_bias, activation)

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.subheader("Input Features")
    st.write(X)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_activations(hidden_activation, "Hidden Layer"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_activations(output_activation, "Output Layer"), use_container_width=True)

    return X, hidden_weights, hidden_bias, output_weights, output_bias, hidden_activation, output_activation

def forward_propagation_calculations(X, hidden_weights, hidden_bias, output_weights, output_bias, hidden_activation, output_activation):
    st.markdown("<h2 class='sub-header'>Forward Propagation Calculations</h2>", unsafe_allow_html=True)

    st.write("Input Features (X):")
    st.write(X)

    st.write("Hidden Layer Weights (W1):")
    st.write(hidden_weights)

    st.write("Hidden Layer Bias (b1):")
    st.write(hidden_bias)

    st.write("Output Layer Weights (W2):")
    st.write(output_weights)

    st.write("Output Layer Bias (b2):")
    st.write(output_bias)

    st.write("Step 1: Calculate the weighted sum of inputs for the hidden layer (z1)")
    z1 = np.dot(X, hidden_weights) + hidden_bias
    st.write("z1 = X · W1 + b1")
    st.write(z1)

    st.write("Step 2: Apply the activation function to the weighted sum (a1)")
    st.write(f"a1 = {st.session_state.params['activation_func']}(z1)")
    st.write(hidden_activation)

    st.write("Step 3: Calculate the weighted sum of inputs for the output layer (z2)")
    z2 = np.dot(hidden_activation, output_weights) + output_bias
    st.write("z2 = a1 · W2 + b2")
    st.write(z2)

    st.write("Step 4: Apply the activation function to the weighted sum (a2)")
    st.write(f"a2 = {st.session_state.params['activation_func']}(z2)")
    st.write(output_activation)

def learning_center():
    st.markdown("<h2 class='sub-header'>Forward Propagation Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Forward propagation is the process of passing the input data through the neural network to obtain the predicted output.
    At each layer, the input data is transformed using the weights and biases, and an activation function is applied to introduce non-linearity.
    The last layer produces the final output of the network, which can be used for prediction or classification.

    <h3>Steps in Forward Propagation:</h3>
    1. The input data is multiplied by the weights of the first layer and the bias is added.
    2. An activation function (e.g., sigmoid or softmax) is applied to the sum.
    3. The result is passed as input to the next layer, and steps 1-2 are repeated until the final layer.
    4. The output of the last layer gives the predictions of the neural network.

    <h3>Example:</h3>
    In this demo, we randomly generate input data and weights for a simple neural network with one hidden layer.
    You can adjust the number of input features, hidden neurons, output neurons, and the activation function using the sidebar controls.
    The activations of the hidden layer and output layer are visualized using bar plots, where each bar represents the activation of a neuron in that layer.
    </p>
    """, unsafe_allow_html=True)

def quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main purpose of forward propagation in a neural network?",
            "options": [
                "To update the weights of the network",
                "To pass input data through the network and get predictions",
                "To calculate the error of the network",
                "To train the neural network"
            ],
            "correct": 1,
            "explanation": "Forward propagation is like sending a message through a chain of people. Each person (neuron) receives the message, modifies it slightly, and passes it on. The final person gives us the network's prediction or output."
        },
        {
            "question": "What does an activation function do in a neural network?",
            "options": [
                "It determines the learning rate of the network",
                "It initializes the weights of the network",
                "It introduces non-linearity to the network",
                "It calculates the loss of the network"
            ],
            "correct": 2,
            "explanation": "An activation function is like a decision-maker in each neuron. It looks at the input and decides how strongly to fire, adding non-linearity. For example, the sigmoid function squishes any input into a value between 0 and 1, like deciding how much to turn on a dimmer switch."
        },
        {
            "question": "In the context of neural networks, what is a 'layer'?",
            "options": [
                "A group of neurons that process information together",
                "The process of updating weights",
                "The final output of the network",
                "The input data to the network"
            ],
            "correct": 0,
            "explanation": "A layer in a neural network is like a team of workers in an assembly line. Each worker (neuron) in the team processes some information and passes it to the next team. For instance, in an image recognition network, one layer might detect edges, the next shapes, and so on."
        },
        {
            "question": "What's the difference between the hidden layer and the output layer?",
            "options": [
                "Hidden layers use activation functions, output layers don't",
                "Hidden layers process intermediate features, output layer gives final predictions",
                "Hidden layers have weights, output layers don't",
                "There is no difference, they function the same way"
            ],
            "correct": 1,
            "explanation": "Think of a hidden layer as the behind-the-scenes work in a detective agency. They gather and process clues (features). The output layer is like the detective presenting the final conclusion. For example, in a network classifying animals, hidden layers might detect features like 'has fur', 'has four legs', while the output layer combines these to predict 'it's a cat'."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a neural network expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering neural networks. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>🧠 Neural Network Forward Propagation Explorer 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

    configure_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Network Visualization", "🧮 Calculations", "📚 Learning Center", "🎓 Quiz"])

    with tab1:
        X, hidden_weights, hidden_bias, output_weights, output_bias, hidden_activation, output_activation = network_visualization()
    
    with tab2:
        forward_propagation_calculations(X, hidden_weights, hidden_bias, output_weights, output_bias, hidden_activation, output_activation)
    
    with tab3:
        learning_center()
    
    with tab4:
        quiz()

if __name__ == "__main__":
    main()
