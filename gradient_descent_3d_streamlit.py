import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Function with multiple local minima
def f(x, y):
    return 2 * np.sin(0.5*x) * np.cos(0.5*y) + (x**2 + y**2) / 20

# Gradient of the function
def gradient(x, y):
    dx = np.cos(0.5*x) * np.cos(0.5*y) + x / 10
    dy = -np.sin(0.5*x) * np.sin(0.5*y) + y / 10
    return np.array([dx, dy])

# Gradient descent step
def gradient_descent_step(current_pos, learning_rate):
    grad = gradient(current_pos[0], current_pos[1])
    return current_pos - learning_rate * grad

# Random restart
def random_restart():
    return np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10)])

# Create 3D surface plot
def create_3d_plot(all_paths, best_pos):
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='viridis')])

    for i, path in enumerate(all_paths):
        path = np.array(path)
        fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=f(path[:, 0], path[:, 1]),
                                   mode='lines+markers', 
                                   line=dict(color='red', width=2),
                                   marker=dict(color='red', size=3),
                                   name=f'Path {i+1}'))

    fig.add_trace(go.Scatter3d(x=[best_pos[0]], y=[best_pos[1]], z=[f(*best_pos)],
                               mode='markers', marker=dict(color='yellow', size=8, symbol='diamond'),
                               name='Global Best'))

    fig.update_layout(title='3D Global Optimization',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      width=800, height=800)

    return fig

# Streamlit app
def main():
    st.title("3D Global Optimization Demo")

    st.sidebar.header("Optimization Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
    num_iterations = st.sidebar.slider("Number of Iterations per Restart", 10, 500, 100)
    num_restarts = st.sidebar.slider("Number of Restarts", 1, 20, 5)

    if st.sidebar.button("Run Optimization"):
        best_pos = None
        best_value = float('inf')
        all_paths = []

        for _ in range(num_restarts):
            current_pos = random_restart()
            path = [current_pos]

            for _ in range(num_iterations):
                current_pos = gradient_descent_step(current_pos, learning_rate)
                path.append(current_pos)

            all_paths.append(path)
            final_value = f(*current_pos)
            if final_value < best_value:
                best_value = final_value
                best_pos = current_pos

        # Create and display the 3D plot
        fig = create_3d_plot(all_paths, best_pos)
        st.plotly_chart(fig)

        st.write(f"Best Position: ({best_pos[0]:.2f}, {best_pos[1]:.2f})")
        st.write(f"Best Value: {best_value:.4f}")

if __name__ == "__main__":
    main()