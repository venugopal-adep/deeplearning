import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron_output(inputs, weights, bias):
    weighted_sum = sum(w * i for w, i in zip(weights, inputs)) + bias
    return weighted_sum, sigmoid(weighted_sum)

def plot_network(inputs, weights, bias, output, error=None):
    fig = go.Figure()
    
    # Input nodes
    y_positions = [2, 1, 0]  # Reversed order to match image
    colors = ['#FF9999', '#66CCCC', '#FF9966']
    
    # Add input nodes
    for i, (y, color) in enumerate(zip(y_positions, colors)):
        fig.add_trace(go.Scatter(
            x=[0], y=[y],
            mode='markers+text',
            marker=dict(size=30, color=color),
            text=f"{inputs[i]:.2f}",
            textposition='middle right',
            name=f"Input {i+1}"
        ))
    
    # Add neuron node
    fig.add_trace(go.Scatter(
        x=[1], y=[1],
        mode='markers',
        marker=dict(size=40, color='#2E8B57'),
        name="Neuron"
    ))
    
    # Add output node
    fig.add_trace(go.Scatter(
        x=[2], y=[1],
        mode='markers+text',
        marker=dict(size=30, color='#2E8B57'),
        text=f"{output:.2f}",
        textposition='middle left',
        name="Output"
    ))
    
    # Add connections with weights
    for i, y in enumerate(y_positions):
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y, 1],
            mode='lines+text',
            line=dict(color=colors[i], width=2),
            text=[f"w{i+1}={weights[i]:.2f}"],
            textposition='middle center',
            showlegend=False
        ))
    
    # Add connection to output
    fig.add_trace(go.Scatter(
        x=[1, 2], y=[1, 1],
        mode='lines',
        line=dict(color='#2E8B57', width=2),
        showlegend=False
    ))
    
    # Add bias and error annotations
    fig.add_annotation(
        x=1, y=1.5,
        text=f"bias={bias:.2f}",
        showarrow=False
    )
    
    if error is not None:
        fig.add_annotation(
            x=1, y=-0.5,
            text=f"Error: {error:.3f}",
            showarrow=False
        )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        xaxis=dict(visible=False, range=[-0.5, 2.5]),
        yaxis=dict(visible=False, range=[-1, 3]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        plot_bgcolor='white'
    )
    
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Neural Network Simulator")
    
    # Initialize session state
    if 'weights' not in st.session_state:
        st.session_state.weights = [0.44, -0.93, 0.81]  # Initial weights from image
        st.session_state.bias = -2.03  # Initial bias from image
        st.session_state.history = []
        st.session_state.iteration = 0

    # Sidebar controls
    with st.sidebar:
        st.header("Network Controls")
        inputs = [
            st.number_input(f"Input {i+1}", -1.0, 1.0, 
                          [-0.20, 0.40, 0.00][i], 0.01, 
                          key=f'input_{i}') 
            for i in range(3)
        ]
        
        learning_rate = st.number_input("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        target = st.selectbox("Target Output", [0, 1])
        
        if st.button("🔄 Reset Network"):
            st.session_state.weights = [0.44, -0.93, 0.81]
            st.session_state.bias = -2.03
            st.session_state.history = []
            st.session_state.iteration = 0
        
        if st.button("⚡ Train One Step"):
            st.session_state.iteration += 1
            weighted_sum, output = neuron_output(inputs, st.session_state.weights, st.session_state.bias)
            error = target - output
            
            # Update weights and bias
            for i in range(len(inputs)):
                st.session_state.weights[i] += learning_rate * error * inputs[i]
            st.session_state.bias += learning_rate * error
            
            # Store training data
            st.session_state.history.append({
                'Iteration': st.session_state.iteration,
                'Target': target,
                'Output': output,
                'Error': error,
                'Weights': st.session_state.weights.copy(),
                'Bias': st.session_state.bias
            })

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("Neural Networks - Forward & Backward Propagation")
        weighted_sum, output = neuron_output(inputs, st.session_state.weights, st.session_state.bias)
        error = None if not st.session_state.history else st.session_state.history[-1]['Error']
        
        # Network visualization
        st.plotly_chart(plot_network(inputs, st.session_state.weights, st.session_state.bias, 
                                   output, error), use_container_width=True)
        
        # Calculations
        st.markdown("### Calculations")
        st.markdown("#### Forward Propagation")
        st.latex(f"z = {' + '.join([f'({w:.2f} × {x:.2f})' for w, x in zip(st.session_state.weights, inputs)])} + {st.session_state.bias:.2f}")
        st.latex(f"output = sigmoid({weighted_sum:.3f}) = {output:.3f}")
        
        if error is not None:
            st.markdown("#### Backward Propagation")
            st.latex(f"error = {target} - {output:.3f} = {error:.3f}")
            for i in range(len(inputs)):
                st.latex(f"Δw_{i+1} = {learning_rate} × {error:.3f} × {inputs[i]:.2f}")
            st.latex(f"Δbias = {learning_rate} × {error:.3f}")
    
    with col2:
        if st.session_state.history:
            st.markdown("### Training Progress")
            df = pd.DataFrame(st.session_state.history)
            
            # Progress plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Iteration'], y=df['Error'].abs(), 
                                   mode='lines', name='|Error|', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df['Iteration'], y=df['Output'], 
                                   mode='lines', name='Output', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Iteration'], y=df['Target'], 
                                   mode='lines', name='Target', line=dict(color='green', dash='dash')))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Training history table
            st.markdown("### Training History")
            display_df = df[['Iteration', 'Target', 'Output', 'Error', 'Bias']].copy()
            for i in range(len(st.session_state.weights)):
                display_df[f'W{i+1}'] = df['Weights'].apply(lambda x: x[i])
            
            st.dataframe(
                display_df.sort_values('Iteration', ascending=False).style.format({
                    'Output': '{:.3f}',
                    'Error': '{:.3f}',
                    'Bias': '{:.3f}',
                    'W1': '{:.3f}',
                    'W2': '{:.3f}',
                    'W3': '{:.3f}'
                }),
                use_container_width=True,
                height=200
            )

if __name__ == "__main__":
    main()
