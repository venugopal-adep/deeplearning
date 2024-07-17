import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(layout="wide", page_title="Cost Functions Explorer", page_icon="💰")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #1E90FF;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
        color: #4682B4;
    }
    .small-font {
        font-size: 16px !important;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4169E1;'>💰 Cost Functions Explorer 💰</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Cost Functions Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's dive into the world of cost functions and see how they impact machine learning models.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Cost Functions?</p>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight'>
<p class='small-font'>
Cost functions are like scorekeepers in machine learning. They measure how well our model is doing by comparing its predictions to the actual values. The goal is to minimize this score, which means our model is making better predictions.

Imagine you're playing darts:
- The bullseye is the perfect prediction
- The cost function measures how far your darts (predictions) are from the bullseye
- The lower the score, the better your aim (model's performance)

We'll explore three common cost functions:
1. Mean Squared Error (MSE): Squares the differences, penalizes larger errors more
2. Mean Absolute Error (MAE): Takes the absolute difference, treats all errors equally
3. Huber Loss: A combination of MSE and MAE, less sensitive to outliers

We'll use these to predict house prices in California and see how they affect our model's performance!
</p>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df

df = load_data()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Data Exploration", "📊 Cost Functions Visualization", "🔮 Model Training", "🧠 Quiz"])

with tab1:
    st.markdown("<p class='medium-font'>California Housing Data Exploration</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        We're using the California Housing dataset. It contains information about different houses in California and their prices.
        Let's explore some key features!
        </p>
        """, unsafe_allow_html=True)

        st.write(f"Dataset shape: {df.shape}")
        st.write("First few rows:")
        st.write(df.head())
        
        feature = st.selectbox("Select a feature to visualize", df.columns)
        
    with col2:
        fig = go.Figure(data=go.Scatter(x=df[feature], y=df['PRICE'], mode='markers'))
        fig.update_layout(
            title=f'{feature} vs House Price',
            xaxis_title=feature,
            yaxis_title="Price ($100,000s)",
            height=400
        )
        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Cost Functions Visualization</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's visualize how different cost functions behave. 
        Adjust the prediction error and see how each cost function responds!
        </p>
        """, unsafe_allow_html=True)

        true_value = 10
        predicted_value = st.slider("Predicted value", 0.0, 20.0, 10.0, 0.1)
        error = predicted_value - true_value

        mse = error ** 2
        mae = abs(error)
        delta = 1.0  # Huber loss delta
        huber = 0.5 * error ** 2 if abs(error) <= delta else delta * (abs(error) - 0.5 * delta)

        st.markdown(f"""
        <p class='small-font'>
        True value: {true_value}<br>
        Predicted value: {predicted_value:.1f}<br>
        Error: {error:.2f}<br><br>
        MSE: {mse:.4f}<br>
        MAE: {mae:.4f}<br>
        Huber Loss: {huber:.4f}
        </p>
        """, unsafe_allow_html=True)
        
    with col2:
        errors = np.linspace(-10, 10, 100)
        mse_values = errors ** 2
        mae_values = np.abs(errors)
        huber_values = np.where(np.abs(errors) <= delta, 
                                0.5 * errors ** 2, 
                                delta * (np.abs(errors) - 0.5 * delta))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=errors, y=mse_values, mode='lines', name='MSE'))
        fig.add_trace(go.Scatter(x=errors, y=mae_values, mode='lines', name='MAE'))
        fig.add_trace(go.Scatter(x=errors, y=huber_values, mode='lines', name='Huber'))
        fig.add_trace(go.Scatter(x=[error], y=[mse], mode='markers', name='Current MSE', marker=dict(size=10, color='red')))
        fig.add_trace(go.Scatter(x=[error], y=[mae], mode='markers', name='Current MAE', marker=dict(size=10, color='green')))
        fig.add_trace(go.Scatter(x=[error], y=[huber], mode='markers', name='Current Huber', marker=dict(size=10, color='blue')))

        fig.update_layout(
            title='Cost Functions Comparison',
            xaxis_title='Error',
            yaxis_title='Cost',
            height=400
        )
        st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Model Training with Different Cost Functions</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Now, let's train models using different cost functions and compare their performance!
        </p>
        """, unsafe_allow_html=True)

        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        if st.button("Train Models"):
            X = df.drop('PRICE', axis=1)
            y = df['PRICE']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            lr_model = LinearRegression().fit(X_train_scaled, y_train)
            ridge_model = Ridge().fit(X_train_scaled, y_train)
            lasso_model = Lasso().fit(X_train_scaled, y_train)
            
            # Make predictions
            lr_pred = lr_model.predict(X_test_scaled)
            ridge_pred = ridge_model.predict(X_test_scaled)
            lasso_pred = lasso_model.predict(X_test_scaled)
            
            # Calculate metrics
            models = {
                "Linear Regression (MSE)": lr_pred,
                "Ridge (L2 Regularization)": ridge_pred,
                "Lasso (L1 Regularization)": lasso_pred
            }
            
            results = []
            for name, pred in models.items():
                mse = mean_squared_error(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                results.append([name, mse, mae, r2])
            
            results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R-squared"])
            st.write(results_df)

    with col2:
        if 'results_df' in locals():
            fig = go.Figure()
            for metric in ["MSE", "MAE"]:
                fig.add_trace(go.Bar(x=results_df["Model"], y=results_df[metric], name=metric))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Error',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig)
            
            fig = go.Figure(data=go.Bar(x=results_df["Model"], y=results_df["R-squared"]))
            fig.update_layout(
                title='Model R-squared Comparison',
                xaxis_title='Model',
                yaxis_title='R-squared',
                height=400
            )
            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the main purpose of a cost function in machine learning?",
            "options": [
                "To increase the model's complexity",
                "To measure how well the model's predictions match the actual values",
                "To decrease the training time",
                "To add more features to the dataset"
            ],
            "correct": 1,
            "explanation": "Cost functions measure the difference between the model's predictions and the actual values, helping us understand how well the model is performing."
        },
        {
            "question": "Which cost function is more sensitive to outliers?",
            "options": [
                "Mean Absolute Error (MAE)",
                "Mean Squared Error (MSE)",
                "Huber Loss",
                "All of the above are equally sensitive"
            ],
            "correct": 1,
            "explanation": "MSE is more sensitive to outliers because it squares the errors, which amplifies the effect of large errors (outliers)."
        },
        {
            "question": "What's the main advantage of using Huber Loss?",
            "options": [
                "It always results in better model performance",
                "It's computationally less expensive",
                "It combines the benefits of MSE and MAE, being less sensitive to outliers",
                "It allows for faster model training"
            ],
            "correct": 2,
            "explanation": "Huber Loss combines the benefits of MSE and MAE. It behaves like MSE for small errors and like MAE for large errors, making it less sensitive to outliers."
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
st.markdown("""
<div class='highlight'>
<p class='small-font'>
You've explored the world of cost functions and their impact on machine learning models using California housing data. Remember these key points:

1. Cost functions are like scorekeepers, measuring how well our model performs.
2. Different cost functions have different strengths and weaknesses:
   - MSE is sensitive to outliers but provides a smooth gradient for optimization.
   - MAE is less sensitive to outliers but can be harder to optimize.
   - Huber Loss combines the benefits of both MSE and MAE.
3. The choice of cost function can significantly impact your model's performance.
4. Regularization techniques like Ridge (L2) and Lasso (L1) can help prevent overfitting.

Keep exploring and experimenting with different cost functions to find the best fit for your machine learning projects!
</p>
</div>
""", unsafe_allow_html=True)

# Add a footnote about the libraries and dataset used
st.markdown("""
<p class='small-font' style='text-align: center; color: gray;'>
This app uses scikit-learn for machine learning models and data handling, and Plotly for interactive visualizations. 
The California Housing dataset is used as an example dataset. These resources make it easier to explore and understand complex machine learning concepts.
</p>
""", unsafe_allow_html=True)