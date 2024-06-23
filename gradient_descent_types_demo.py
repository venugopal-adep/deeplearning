import pygame
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gradient Descent Methods Comparison")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Fonts
font = pygame.font.Font(None, 24)
title_font = pygame.font.Font(None, 36)

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=20)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Scale to fit screen
X = (X - X.min()) / (X.max() - X.min()) * (WIDTH * 0.6)  # Use only 60% of width for data
y = (y - y.min()) / (y.max() - y.min()) * (HEIGHT * 0.6)  # Use only 60% of height for data
data_points = np.column_stack((X, y))

# Model parameters
params = {method: np.array([0.0, 0.0]) for method in ['batch', 'stochastic', 'mini_batch']}

# Hyperparameters
learning_rates = {'batch': 0.01, 'stochastic': 0.001, 'mini_batch': 0.005}
mini_batch_size = 10
epochs = 0

def draw_data_points():
    for point in data_points:
        pygame.draw.circle(screen, BLACK, (int(point[0]) + 50, int(point[1]) + 50), 3)

def draw_line(params, color):
    y1 = params[0] + params[1] * 0
    y2 = params[0] + params[1] * (WIDTH * 0.6)
    y1 = (y1 - y.min()) / (y.max() - y.min()) * (HEIGHT * 0.6)
    y2 = (y2 - y.min()) / (y.max() - y.min()) * (HEIGHT * 0.6)
    pygame.draw.line(screen, color, (50, y1 + 50), (WIDTH * 0.6 + 50, y2 + 50), 2)

def mean_squared_error(params):
    y_pred = params[0] + params[1] * X.flatten()
    return np.mean(np.square(y - y_pred))

def compute_gradients(params, X_batch, y_batch):
    m = len(X_batch)
    y_pred = params[0] + params[1] * X_batch
    dw = -2/m * np.sum(X_batch * (y_batch - y_pred))
    db = -2/m * np.sum(y_batch - y_pred)
    return np.array([db, dw])

def gradient_descent(params, method):
    if method == 'batch':
        grads = compute_gradients(params, X.flatten(), y)
    elif method == 'stochastic':
        idx = np.random.randint(0, len(X))
        grads = compute_gradients(params, X[idx], y[idx])
    else:  # mini_batch
        indices = np.random.choice(len(X), mini_batch_size, replace=False)
        grads = compute_gradients(params, X[indices], y[indices])
    
    grads = np.clip(grads, -1, 1)
    return params - learning_rates[method] * grads

methods = ['batch', 'stochastic', 'mini_batch']
colors = {'batch': RED, 'stochastic': GREEN, 'mini_batch': BLUE}
errors = {method: [] for method in methods}

running = True
paused = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                params = {method: np.array([0.0, 0.0]) for method in methods}
                errors = {method: [] for method in methods}
                epochs = 0

    if not paused:
        screen.fill(WHITE)
        
        # Draw data area border
        pygame.draw.rect(screen, GRAY, (40, 40, WIDTH * 0.6 + 20, HEIGHT * 0.6 + 20), 2)
        
        draw_data_points()

        for method in methods:
            params[method] = gradient_descent(params[method], method)
            draw_line(params[method], colors[method])
            error = mean_squared_error(params[method])
            errors[method].append(error)

        epochs += 1

        # Display title
        title = title_font.render("Gradient Descent Methods Comparison", True, BLACK)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 10))

        # Display information
        info_text = [
            f"Epochs: {epochs}",
            f"Batch GD Error: {errors['batch'][-1]:.4f}",
            f"Stochastic GD Error: {errors['stochastic'][-1]:.4f}",
            f"Mini-Batch GD Error: {errors['mini_batch'][-1]:.4f}",
            "Space: Pause/Resume | R: Reset"
        ]

        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (WIDTH * 0.7, 50 + i * 30))

        # Plot error curves
        if len(errors['batch']) > 1:
            max_error = max(max(errors[m]) for m in methods)
            min_error = min(min(errors[m]) for m in methods)
            for method in methods:
                points = [(WIDTH * 0.7 + i, HEIGHT * 0.5 - ((e - min_error) / (max_error - min_error)) * (HEIGHT * 0.3)) 
                          for i, e in enumerate(errors[method][-int(WIDTH*0.3):])]
                if len(points) > 1:
                    pygame.draw.lines(screen, colors[method], False, points, 2)

        # Draw legends
        legends = [
            ("Batch Gradient Descent", RED),
            ("Stochastic Gradient Descent", GREEN),
            ("Mini-Batch Gradient Descent", BLUE)
        ]
        for i, (text, color) in enumerate(legends):
            pygame.draw.line(screen, color, (WIDTH * 0.7, HEIGHT * 0.7 + i * 30), (WIDTH * 0.7 + 50, HEIGHT * 0.7 + i * 30), 2)
            legend_text = font.render(text, True, BLACK)
            screen.blit(legend_text, (WIDTH * 0.7 + 60, HEIGHT * 0.7 + i * 30 - 10))

        pygame.display.flip()
        pygame.time.delay(10)

pygame.quit()