import streamlit as st
import numpy as np
import plotly.graph_objects as go

def convex_func(x):
    return (x - 2)**2 + 1

def non_convex_func(x):
    return np.sin(x) + 0.5*x

def plot_function(x, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Function'))
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='markers', marker=dict(color='red', size=10), name='Current Point'))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='f(x)')
    return fig

def main():
    st.title("Optimization Demo")
    st.write("**Developed by : Venugopal Adep**")
    
    st.header("Convex Optimization")
    st.write("Convex optimization involves a function that has only one minimum, corresponding to the global optimum. There is no concept of local optima for convex optimization problems.")
    
    x = np.linspace(-5, 5, 100)
    x_slider_convex = st.slider("x (Convex)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    y_convex = convex_func(x)
    fig_convex = plot_function(x, y_convex, 'Convex Function')
    fig_convex.data[1].x = [x_slider_convex]
    fig_convex.data[1].y = [convex_func(x_slider_convex)]
    st.plotly_chart(fig_convex)
    
    st.header("Non-Convex Optimization")
    st.write("Non-convex optimization involves a function that has multiple local optima. Finding the global optimum in non-convex optimization can be challenging.")
    
    x_slider_non_convex = st.slider("x (Non-Convex)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    y_non_convex = non_convex_func(x)
    fig_non_convex = plot_function(x, y_non_convex, 'Non-Convex Function')
    fig_non_convex.data[1].x = [x_slider_non_convex]
    fig_non_convex.data[1].y = [non_convex_func(x_slider_non_convex)]
    st.plotly_chart(fig_non_convex)
    
    st.write("Observe how the convex function has a single global minimum, while the non-convex function has multiple local optima. The red dot indicates the current x-value selected by the slider.")

if __name__ == '__main__':
    main()
