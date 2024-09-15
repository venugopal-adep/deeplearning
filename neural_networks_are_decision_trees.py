import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_moons

# Set page config
st.set_page_config(page_title="Neural Networks as Decision Trees", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stTab {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4682b4;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #4682b4;
    }
    body {
        color: #333;
        background-color: #f0f8ff;
    }
    h1, h2, h3 {
        color: #4682b4;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate data for y = x^2 regression
def generate_x_squared_data(n_samples=500):
    X = np.linspace(-2.5, 2.5, n_samples).reshape(-1, 1)
    y = X**2
    return X, y

# Function to plot regression results
def plot_regression(X, y, y_pred_nn, y_pred_tree):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='True Data', marker=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred_nn.flatten(), mode='lines', name='Neural Network', line=dict(color='#ff7f0e', width=3)))
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred_tree.flatten(), mode='lines', name='Decision Tree', line=dict(color='#2ca02c', width=3)))
    fig.update_layout(
        title='y = x^2 Regression',
        xaxis_title='x',
        yaxis_title='y',
        plot_bgcolor='rgba(240,248,255,0.5)',
        paper_bgcolor='rgba(240,248,255,0.5)'
    )
    return fig

# Function to plot classification results
def plot_classification(X, y, model_nn, model_tree):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z_nn = model_nn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Z_tree = model_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z_nn, colorscale='Blues', opacity=0.5, name='Neural Network'))
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z_tree, colorscale='Greens', opacity=0.5, name='Decision Tree'))
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', marker=dict(color='#ff7f0e', symbol='circle', size=8), name='Class 0'))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', marker=dict(color='#e377c2', symbol='x', size=8), name='Class 1'))
    fig.update_layout(
        title='Half-Moon Classification',
        xaxis_title='x',
        yaxis_title='y',
        plot_bgcolor='rgba(240,248,255,0.5)',
        paper_bgcolor='rgba(240,248,255,0.5)'
    )
    return fig

# Streamlit app
st.title("Neural Networks as Decision Trees: An Interactive Exploration")
st.write("**Developed by : Venugopal Adep**")
st.write("Discover how complex neural networks can be understood as simple decision trees!")

tab1, tab2, tab3 = st.tabs(["Introduction", "Regression Demo", "Classification Demo"])

with tab1:
    st.header("Understanding Neural Networks and Decision Trees")
    st.write("""
    Imagine you're trying to teach a computer to recognize cats and dogs in pictures. You have two options:

    1. **Neural Network**: Think of this as a complex system of interconnected 'brain cells'. 
       It learns by looking at thousands of cat and dog pictures, gradually figuring out what makes a cat a cat and a dog a dog.

    2. **Decision Tree**: This is like a flowchart of yes/no questions. 
       "Does it have pointy ears? Does it bark? Does it meow?" Based on the answers, it decides if it's a cat or a dog.

    Now, here's the fascinating part: Even though neural networks seem much more complex, 
    researchers have found that any neural network can be represented as a decision tree!

    In this app, we'll explore this idea with simple examples. Let's dive in!
    """)

with tab2:
    st.header("Regression Example: Predicting a Curve")
    st.write("""
    Let's start with a simple problem: predicting points on a curve. 
    We'll use the equation y = x², which creates a U-shaped curve.

    We'll train both a neural network and a decision tree to predict this curve, 
    and compare their performances.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        n_samples = st.slider("Number of data points", 100, 1000, 500, 100, key="reg_n_samples")
        noise_level = st.slider("Noise level", 0.0, 0.5, 0.0, 0.05, key="reg_noise")
        max_depth = st.slider("Decision Tree Max Depth", 1, 10, 5, key="reg_max_depth")

    with col2:
        X, y = generate_x_squared_data(n_samples)
        y += np.random.normal(0, noise_level, y.shape)  # Add noise

        nn_regressor = MLPRegressor(hidden_layer_sizes=(3, 3), activation='relu', random_state=42)
        nn_regressor.fit(X, y)

        tree_regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree_regressor.fit(X, y)

        y_pred_nn = nn_regressor.predict(X)
        y_pred_tree = tree_regressor.predict(X)

        st.plotly_chart(plot_regression(X, y, y_pred_nn, y_pred_tree), use_container_width=True)

    st.write("""
    What's happening here?
    - The blue dots are the actual data points.
    - The orange line is the neural network's prediction.
    - The green line is the decision tree's prediction.

    Try adjusting the sliders and see how the predictions change!

    Notice how both the neural network and the decision tree try to fit the curve. 
    The neural network usually gives a smoother prediction, while the decision tree's prediction 
    looks more like a series of connected straight lines. This is because the decision tree 
    is making a series of yes/no decisions to arrive at its prediction.
    """)

with tab3:
    st.header("Classification Example: Separating Two Groups")
    st.write("""
    Now, let's try a classification problem. Imagine we have two intertwined groups of points, 
    shaped like half-moons. We want our models to learn to separate these two groups.

    This is similar to many real-world problems, like separating spam from non-spam emails, 
    or deciding whether a customer is likely to buy a product or not.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        n_samples = st.slider("Number of data points", 100, 500, 200, 50, key="class_n_samples")
        noise_level = st.slider("Noise level", 0.1, 0.5, 0.3, 0.05, key="class_noise")
        max_depth = st.slider("Decision Tree Max Depth", 1, 10, 5, key="class_max_depth")

    with col2:
        X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)

        nn_classifier = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', random_state=42)
        nn_classifier.fit(X, y)

        tree_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree_classifier.fit(X, y)

        st.plotly_chart(plot_classification(X, y, nn_classifier, tree_classifier), use_container_width=True)

    st.write("""
    What's happening here?
    - The orange circles and pink X's are the two groups we're trying to separate.
    - The background colors show how each model is dividing the space:
        - Blue areas show the neural network's decision boundary.
        - Green areas show the decision tree's boundary.

    Try adjusting the sliders and see how the models adapt!

    Notice how the neural network often creates a smoother boundary, while the decision tree's 
    boundary is made up of straight lines (because it's making a series of yes/no decisions).

    Despite these differences, the amazing thing is that we can always represent the neural network's 
    complex decision process as a (potentially very large) decision tree!
    """)

st.header("Key Takeaways")
st.write("""
1. Any neural network can be represented as a decision tree, no matter how complex!
2. This representation is exact, not an approximation.
3. This helps us understand how neural networks make decisions.
4. For simple problems, decision trees might be easier to understand and just as effective.
5. For complex problems, neural networks might perform better, but we can still analyze them as very large decision trees.

Remember, this demo uses simple examples, but the principle extends to much more complex neural networks too!
""")

# To run this app, save it as a .py file and run: streamlit run filename.py
