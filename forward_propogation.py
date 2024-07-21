import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Custom CSS and Page Configuration
st.set_page_config(layout="wide", page_title="Ensemble Learning Explorer", page_icon="🌳")

st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #4B0082; text-align: center; margin-bottom: 2rem; text-shadow: 2px 2px 4px #cccccc;}
    .sub-header {font-size: 2rem; color: #8A2BE2; margin: 1.5rem 0;}
    .content-text {font-size: 1.1rem; line-height: 1.6;}
    .highlight {background-color: #E6E6FA; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;}
    .interpretation {background-color: #F0E6FF; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 5px solid #8A2BE2;}
    .quiz-question {background-color: #F0E6FF; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 5px solid #8A2BE2;}
    .explanation {background-color: #E6E6FA; padding: 0.8rem; border-radius: 5px; margin-top: 0.8rem;}
    .stButton>button {
        background-color: #9370DB; color: white; font-size: 1rem; padding: 0.5rem 1rem;
        border: none; border-radius: 4px; cursor: pointer; transition: all 0.3s;
    }
    .stButton>button:hover {background-color: #8A2BE2; transform: scale(1.05);}
</style>
""", unsafe_allow_html=True)

# Main Application
def main():
    st.markdown("<h1 class='main-header'>🌳 Ensemble Learning Explorer: Bagging vs Random Forest 🌳</h1>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

    # Sidebar Configuration
    configure_sidebar()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Model Visualizer", "📊 Performance Analysis", "🎓 Learning Center", "🧠 Knowledge Quiz"])

    with tab1:
        model_visualizer()
    with tab2:
        performance_analysis()
    with tab3:
        learning_center()
    with tab4:
        knowledge_quiz()

    conclusion()

def configure_sidebar():
    st.sidebar.title("Model Parameters")
    params = {
        'n_estimators': st.sidebar.slider("Number of Estimators", 1, 100, 10),
        'n_samples': st.sidebar.slider("Number of Samples", 100, 1000, 500, 50),
        'noise': st.sidebar.slider("Noise Level", 0.0, 0.5, 0.3, 0.1)
    }
    return params

def generate_data(params):
    X, y = make_moons(n_samples=params['n_samples'], noise=params['noise'], random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train, params):
    bagging = BaggingClassifier(n_estimators=params['n_estimators'], random_state=42)
    rf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=42)
    
    bagging.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    return bagging, rf

def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', opacity=0.5, showscale=False, contours=dict(start=0, end=1, size=0.5)))
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(size=10, color=y, colorscale='RdBu', line=dict(color='Black', width=1)), showlegend=False))
    
    fig.update_layout(title=title, xaxis_title='Feature 1', yaxis_title='Feature 2', width=700, height=600, autosize=False, margin=dict(l=50, r=50, b=50, t=50, pad=4))
    return fig

def model_visualizer():
    st.markdown("<h2 class='sub-header'>Model Visualization</h2>", unsafe_allow_html=True)
    
    params = configure_sidebar()
    X_train, X_test, y_train, y_test = generate_data(params)
    bagging, rf = train_models(X_train, y_train, params)
    
    bagging_accuracy = accuracy_score(y_test, bagging.predict(X_test))
    rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_decision_boundary(bagging, X_train, y_train, f"Bagging (Accuracy: {bagging_accuracy:.2f})"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_decision_boundary(rf, X_train, y_train, f"Random Forest (Accuracy: {rf_accuracy:.2f})"), use_container_width=True)
    
    st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
    st.markdown("""
    <p class='content-text'>
    <strong>Interpretation:</strong>
    - Colored regions represent decision boundaries for class separation.
    - Circles are data points, with colors indicating their true class.
    - More complex boundaries suggest higher model flexibility.
    - Increasing estimators typically smooths decision boundaries.
    - Random Forest often outperforms Bagging due to feature subsampling.
    - Accuracy scores quantify model performance on the test set.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def performance_analysis():
    st.markdown("<h2 class='sub-header'>Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    params = configure_sidebar()
    X_train, X_test, y_train, y_test = generate_data(params)
    bagging, rf = train_models(X_train, y_train, params)
    
    bagging_accuracy = accuracy_score(y_test, bagging.predict(X_test))
    rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='content-text'>Bagging Accuracy: {bagging_accuracy:.3f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='content-text'>Random Forest Accuracy: {rf_accuracy:.3f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    accuracy_data = pd.DataFrame({'Model': ['Bagging', 'Random Forest'], 'Accuracy': [bagging_accuracy, rf_accuracy]})
    fig = px.bar(accuracy_data, x='Model', y='Accuracy', title='Model Accuracy Comparison', color='Model', color_discrete_map={'Bagging': '#9370DB', 'Random Forest': '#8A2BE2'})
    st.plotly_chart(fig, use_container_width=True)

def learning_center():
    st.markdown("<h2 class='sub-header'>Ensemble Learning: Bagging and Random Forest</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Bagging (Bootstrap Aggregating) and Random Forest are ensemble learning methods that combine multiple decision trees to enhance prediction accuracy and mitigate overfitting.

    <h3>Bagging:</h3>
    - Creates multiple subsets of the training data through bootstrap sampling.
    - Trains independent decision trees on each subset.
    - Aggregates predictions (majority voting for classification, averaging for regression).

    <h3>Random Forest:</h3>
    - Extends bagging by introducing feature subsampling at each tree split.
    - Reduces correlation between trees and improves ensemble diversity.

    <h3>Key Differences:</h3>
    1. Feature Selection: Bagging uses all features; Random Forest selects random feature subsets.
    2. Diversity: Random Forest achieves greater tree diversity due to feature subsampling.
    3. Performance: Random Forest often outperforms Bagging, especially on high-dimensional datasets.
    4. Interpretability: Both are less interpretable than single trees, but Random Forest provides feature importance measures.

    Experiment with the interactive demo to observe how these models behave with different parameters!
    </p>
    """, unsafe_allow_html=True)

def knowledge_quiz():
    st.markdown("<h2 class='sub-header'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What key feature distinguishes Random Forest from Bagging?",
            "options": [
                "Use of decision trees",
                "Feature subsampling at each split",
                "Exclusive use in classification tasks",
                "Implementation of boosting"
            ],
            "correct": 1,
            "explanation": "Random Forest introduces additional randomness through feature subsampling at each split, while Bagging uses all features."
        },
        {
            "question": "What does 'bootstrap' mean in the context of Bagging?",
            "options": [
                "Model parameter initialization",
                "Feature scaling technique",
                "Random sampling with replacement",
                "Decision tree pruning method"
            ],
            "correct": 2,
            "explanation": "In Bagging, 'bootstrap' refers to creating multiple training data subsets through random sampling with replacement."
        },
        {
            "question": "How do Bagging and Random Forest typically compare in performance?",
            "options": [
                "Bagging consistently outperforms Random Forest",
                "Random Forest often excels, especially with high-dimensional data",
                "They always perform identically",
                "Bagging is superior for small datasets, Random Forest for large ones"
            ],
            "correct": 1,
            "explanation": "Random Forest often outperforms Bagging, particularly on high-dimensional datasets, due to the added randomness from feature subsampling."
        },
        {
            "question": "How does increasing the number of estimators (trees) affect the decision boundary?",
            "options": [
                "It becomes more complex and irregular",
                "It becomes smoother and more stable",
                "It remains unchanged",
                "It always becomes linear"
            ],
            "correct": 1,
            "explanation": "As the number of estimators increases, decision boundaries typically become smoother and more stable due to the aggregation of multiple trees' predictions."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Bagging and Random Forest expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Bagging and Random Forest. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

def conclusion():
    st.markdown("<h2 class='sub-header'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='content-text'>
    Key takeaways from your exploration of Bagging and Random Forest:

    1. Both are ensemble methods combining multiple decision trees.
    2. Random Forest's feature subsampling often leads to superior performance.
    3. Increasing estimators generally improves model stability, with diminishing returns.
    4. Choose between Bagging and Random Forest based on your specific dataset and problem.
    5. Experiment with parameters to optimize model performance and decision boundaries.

    Continue exploring to build more robust and efficient machine learning models!
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
