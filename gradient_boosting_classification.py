import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.inspection import permutation_importance

# Set page config
st.set_page_config(layout="wide", page_title="Gradient Boosting Classification Explorer", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .small-font {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🌳 Gradient Boosting Classification Explorer 🌳</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Gradient Boosting Classification Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the power of Gradient Boosting for classification tasks.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What is Gradient Boosting?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
Gradient Boosting is a machine learning technique for regression and classification problems. Key points:

- It builds an ensemble of weak learners (typically decision trees) in a stage-wise fashion
- Each new model is trained to correct the errors of the previous models
- It can handle complex non-linear relationships and interactions between features
- Prone to overfitting if not properly regularized
- Widely used in various domains due to its high predictive power

Gradient Boosting is known for its high performance and is often used in winning solutions for machine learning competitions.
</p>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Model Training", "🎛️ Hyperparameter Tuning", "📊 Feature Importance", "🧠 Quiz"])

# Load data
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Iris":
        data = load_iris()
    else:
        raise ValueError("Unknown dataset")
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

with tab1:
    st.markdown("<p class='medium-font'>Gradient Boosting Model Training</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train a Gradient Boosting model on a selected dataset and evaluate its performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"])
        X, y = load_data(dataset)
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        n_estimators = st.slider("Number of estimators", 10, 500, 100, 10)
        learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01)
        max_depth = st.slider("Max depth", 1, 10, 3, 1)
        
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
            
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)
            
            st.markdown(f"""
            <p class='small-font'>
            Train Accuracy: {train_accuracy:.4f}<br>
            Test Accuracy: {test_accuracy:.4f}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'model' in locals():
            # Confusion Matrix
            cm = confusion_matrix(y_test, test_preds)
            fig = ff.create_annotated_heatmap(cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'])
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig)
            
            # ROC Curve
            if len(np.unique(y)) == 2:  # Binary classification
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Hyperparameter Tuning</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's explore how different hyperparameters affect the model's performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="hp_dataset")
        X, y = load_data(dataset)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        n_estimators = st.slider("Number of estimators", 10, 500, 100, 10, key="hp_n_estimators")
        learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01, key="hp_lr")
        max_depth = st.slider("Max depth", 1, 10, 3, 1, key="hp_max_depth")
        min_samples_split = st.slider("Min samples split", 2, 20, 2, 1)
        
        if st.button("Train and Evaluate"):
            results = []
            for n_est in range(10, n_estimators+1, 10):
                model = GradientBoostingClassifier(
                    n_estimators=n_est,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                train_accuracy = accuracy_score(y_train, model.predict(X_train))
                test_accuracy = accuracy_score(y_test, model.predict(X_test))
                results.append((n_est, train_accuracy, test_accuracy))
            
            results_df = pd.DataFrame(results, columns=['n_estimators', 'Train Accuracy', 'Test Accuracy'])
            st.dataframe(results_df)

    with col2:
        if 'results_df' in locals():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['n_estimators'], y=results_df['Train Accuracy'], mode='lines+markers', name='Train Accuracy'))
            fig.add_trace(go.Scatter(x=results_df['n_estimators'], y=results_df['Test Accuracy'], mode='lines+markers', name='Test Accuracy'))
            fig.update_layout(
                title='Model Performance vs Number of Estimators',
                xaxis_title='Number of Estimators',
                yaxis_title='Accuracy'
            )
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Feature Importance</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's examine which features are most important for our Gradient Boosting model.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="fi_dataset")
        X, y = load_data(dataset)
        
        n_estimators = st.slider("Number of estimators", 10, 500, 100, 10, key="fi_n_estimators")
        learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01, key="fi_lr")
        max_depth = st.slider("Max depth", 1, 10, 3, 1, key="fi_max_depth")
        
        if st.button("Calculate Feature Importance"):
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            
            model.fit(X, y)
            
            feature_importance = model.feature_importances_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            st.dataframe(importance_df)

    with col2:
        if 'importance_df' in locals():
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h'
            ))
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=800
            )
            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the main idea behind Gradient Boosting?",
            "options": [
                "Building a single strong learner",
                "Combining weak learners to create a strong learner",
                "Using only the best features for prediction",
                "Reducing the number of features in the dataset"
            ],
            "correct": 1,
            "explanation": "Gradient Boosting builds an ensemble of weak learners (typically decision trees) in a stage-wise fashion, where each new model is trained to correct the errors of the previous models."
        },
        {
            "question": "How does increasing the number of estimators typically affect a Gradient Boosting model?",
            "options": [
                "It always improves both training and test accuracy",
                "It always reduces both training and test accuracy",
                "It typically improves training accuracy but may lead to overfitting",
                "It has no effect on model performance"
            ],
            "correct": 2,
            "explanation": "Increasing the number of estimators typically improves training accuracy but may lead to overfitting if the number becomes too large."
        },
        {
            "question": "What is the role of the learning rate in Gradient Boosting?",
            "options": [
                "It determines the number of trees in the model",
                "It controls the contribution of each tree to the final prediction",
                "It sets the maximum depth of each tree",
                "It determines the number of features to use"
            ],
            "correct": 1,
            "explanation": "The learning rate in Gradient Boosting controls the contribution of each tree to the final prediction. A smaller learning rate means each tree contributes less, requiring more trees to achieve the same performance."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<p class='small-font'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.info(q['explanation'])
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='big-font'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()

# Conclusion
st.markdown("<p class='big-font'>Congratulations! 🎊</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>You've explored Gradient Boosting classification through interactive examples and visualizations. Gradient Boosting is a powerful technique for various machine learning tasks. Keep exploring and applying these concepts to solve real-world problems!</p>", unsafe_allow_html=True)